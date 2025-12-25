# Evaluation Usage Guide

## Environment Dependencies

- Python 3.10+
- Install MTEB: `pip install mteb==1.38.60`
- Install tabulate (for table display): `pip install "tabulate>=0.9.0"`

## MTEB Evaluation

Use `mteb_runner.py` to evaluate the model on the MTEB benchmark. An example command is shown below:

```bash
python mteb_runner.py \
  --model-path /path/to/ckpt.pth \
  --vision-tower-path /path/to/vision_tower \
  --benchmark_name MTEB_ENG_V2 \
  --batch-size 8 \
  --ctx-len 1024 \
  --n-layer 12 \
  --n-embd 768
```

### Bash Script

The repository provides a script to simplify the above process:

```bash
bash scripts/run_mteb.sh /path/to/ckpt.pth MTEB_ENG_V2 cuda:1
```

These scripts demonstrate how to configure model evaluation on different datasets and can be modified as needed.
