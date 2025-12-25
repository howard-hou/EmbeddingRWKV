import os, math, time, datetime, subprocess
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only


def _collect_trainable_layer_modules(pl_module: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers: List[Tuple[str, nn.Module]] = []

    def add_layer(name: str, module: nn.Module):
        if module is None:
            return
        if any(p.requires_grad for p in module.parameters(recurse=True)):
            layers.append((name, module))

    if hasattr(pl_module, "rwkv"):
        add_layer("rwkv.emb", getattr(pl_module.rwkv, "emb", None))
        blocks = getattr(pl_module.rwkv, "blocks", [])
        for idx, block in enumerate(blocks):
            add_layer(f"rwkv.blocks.{idx}", block)
    add_layer("head", getattr(pl_module, "head", None))
    return layers


def build_layerwise_lr_map(pl_module: nn.Module, args) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    trainable_layers = _collect_trainable_layer_modules(pl_module)
    if not trainable_layers:
        return {}, []

    scale_factor = getattr(args, "lr_layer_decay", 1.0)
    num_layers = len(trainable_layers)
    base_lr = args.lr_init / (scale_factor ** max(num_layers - 1, 0)) if scale_factor != 0 else args.lr_init
    head_lr_scale = getattr(args, "lr_head_scale", 5.0)

    layer_lr_map: Dict[str, float] = {}
    layer_lr_summary: List[Tuple[str, float]] = []
    for layer_idx, (layer_name, module) in enumerate(trainable_layers):
        layer_lr = base_lr * (scale_factor ** layer_idx)
        if layer_name.startswith("head"):
            layer_lr *= head_lr_scale
        layer_lr_summary.append((layer_name, layer_lr))
        for param_name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            full_name = f"{layer_name}.{param_name}" if param_name else layer_name
            layer_lr_map[full_name] = layer_lr
    return layer_lr_map, layer_lr_summary

def my_save(args, trainer, dd, ff):
    if 'deepspeed_stage_3' in args.strategy:
        trainer.save_checkpoint(ff, weights_only=True)
    else:
        torch.save(dd, ff)

class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.micro_step = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # global_step is update step, influence by gradient accumulation
        real_step = trainer.global_step * args.accumulate_grad_batches + args.epoch_begin * args.epoch_steps

        # LR schedule, cosine with warmup
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_total = (args.epoch_begin + args.epoch_count) * args.epoch_steps
            progress = (real_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            # cosine decay
            cosine_decay = max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
            lr = args.lr_final + (args.lr_init - args.lr_final) * cosine_decay

        if real_step < w_step:
            lr = lr * (0.1 + 0.9 * real_step / w_step)

        wd_now = args.weight_decay
        optimizer = trainer.optimizers[0]
        lr_scale = lr / args.lr_init
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * lr_scale
            if param_group.get("weight_decay", 0) > 0:
                param_group["weight_decay"] = wd_now

        trainer.my_lr = lr
        trainer.my_wd = wd_now
        # rank_zero_info(f"{real_step} {lr}")

        if trainer.global_step == 0 and self.micro_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb
        self.micro_step += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        sample_per_step = args.real_bsz
        # micro_step is the step of real batch
        real_step = self.micro_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            sample_per_second = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                sample_per_second = sample_per_step / t_cost 
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("sample/s", sample_per_second, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                trainer.my_loss = outputs["loss"]
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            # self.log("s", real_step, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Ksamples": real_step * sample_per_step / 1e3}
                if sample_per_second > 0:
                    lll["sample/s"] = sample_per_second
                trainer.my_wandb.log(lll, step=int(real_step))
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size

    def on_train_epoch_end(self, trainer, pl_module):
        def get_epoch_save_condition(args, trainer):
            # not save first epoch, only if epoch count == 1, save it
            # or epoch_save == 1, always save
            if trainer.current_epoch % args.epoch_save == 0:
                if trainer.current_epoch == 0:
                    if args.epoch_count == 1:
                        return True
                    elif args.epoch_save == 1:
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                return False
    
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and get_epoch_save_condition(args, trainer)) or (trainer.current_epoch == args.epoch_count - 1):
                to_save_dict = pl_module.state_dict()
                try:
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth",
                    )
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0