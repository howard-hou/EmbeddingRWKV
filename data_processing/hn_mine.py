import os
import argparse
import logging
import json
import random
import math
import importlib.util
import numpy as np
from typing import cast, List, Union, Dict, Any
from dataclasses import dataclass, field

from tqdm import tqdm
from transformers import HfArgumentParser
from vllm import LLM

logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.worker").setLevel(logging.ERROR)


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s  %(filename)s : %(levelname)s  %(message)s", 
    datefmt="%Y-%m-%d %A %H:%M:%S") 
logger = logging.getLogger(__name__)


@dataclass
class HNMineArguments:
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to the model for hard negative mining."}
    )
    pooling_method: str = field(
        default="cls", 
        metadata={"help": "The pooling method of the embedding model."}
    )
    input_file: str = field(
        default=None, 
        metadata={"help": "The input file path for hard negative mining."}
    )
    candidate_pool: str = field(
        default=None, 
        metadata={"help": "The candidate document file path for negative pool."}
    )
    output_file: str = field(
        default=None, 
        metadata={"help": "The output file path for saving."}
    )
    range_for_sampling: str = field(
        default="10-210", 
        metadata={"help": "The negative sampling range, e.g. 10-100"}
    )
    use_gpu_for_searching: bool = field(
        default=True,
        metadata={"help": "Whether to use GPU accelerated cuVS searching. Requires RAPIDS cuVS dependencies."}
    )
    negative_number: int = field(
        default=15, 
        metadata={"help": "The number of mined negative samples."}
    )
    filter_topk: int = field(
        default=None, 
        metadata={"help": "The top-k threshold for ranking consistency filtering."}
    )
    query_instruction_for_retrieval: str = field(
        default="", 
        metadata={"help": "The query instruction for retrieval."}
    )
    passages_instruction_for_retrieval: str = field(
        default="", 
        metadata={"help": "The passage instruction for retrieval."}
    )
    batch_size: int = field(
        default=256, 
        metadata={"help": "The batch size for model encoding."}
    )


class vLLMModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            passages_instruction_for_retrieval: str = None,
            use_fp16: bool = False
    ) -> None:
        self.model = LLM(model=model_name_or_path, task="embed", gpu_memory_utilization=0.2)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.passages_instruction_for_retrieval = passages_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method

        # ``use_fp16`` and ``pooling_method`` are kept for backwards compatibility with the
        # previous HuggingFace Transformers implementation. The pooling logic is handled inside
        # vLLM, so these options are currently unused but preserved to avoid breaking callers.
    
    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int=256,
                       max_length: int=32768) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.query_instruction_for_retrieval + queries
            else:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length)
    
    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int=256,
                      max_length: int=32768) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        if there is a instruction for corpus, we will add it to the corpus text
        '''
        if self.passages_instruction_for_retrieval is not None:
            if isinstance(corpus, str):
                input_texts = self.passages_instruction_for_retrieval + corpus
            else:
                input_texts = ['{}{}'.format(self.passages_instruction_for_retrieval, q) for q in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length)


    def encode(self, sentences: Union[List[str], str], batch_size: int=256, max_length: int=32768) -> np.ndarray:
        # ``max_length`` is retained for API compatibility but not used by vLLM embedding calls.
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        # truncate too long sentences
        truncated_sentences = []
        for sentence in sentences:
            if len(sentence) > max_length:
                sentence = sentence[:max_length]
            truncated_sentences.append(sentence)
        sentences = truncated_sentences

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings", disable=len(sentences)<256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            outputs = self.model.embed(sentences_batch)
            embeddings = np.array(
                [cast(List[float], output.outputs.embedding) for output in outputs],
                dtype=np.float32
            )
            if self.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings


def _should_use_gpu_for_search(request_gpu: bool) -> bool:
    if not request_gpu:
        return False

    required_modules = {
        module: importlib.util.find_spec(module) is not None
        for module in ("cupy", "pylibraft", "cuvs")
    }
    missing = [module for module, available in required_modules.items() if not available]
    if missing:
        logger.warning(
            "cuVS search requested but missing dependencies: %s. Falling back to CPU search.",
            ", ".join(missing)
        )
        return False

    import cupy as cp  # type: ignore[import]

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        logger.warning("cuVS search requested but no CUDA-enabled GPU is available. Falling back to CPU search.")
        return False

    if device_count <= 0:
        logger.warning("cuVS search requested but GPU count is 0. Falling back to CPU search.")
        return False

    return True


def create_index(embeddings, use_gpu: bool) -> Dict[str, Any]:
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if use_gpu:
        import cupy as cp  # type: ignore[import]
        import pylibraft  # type: ignore[import]
        from cuvs.neighbors import ivf_flat  # type: ignore[import]

        pylibraft.config.set_output_as(lambda device_ndarray: device_ndarray.copy_to_host())

        device_embeddings = cp.asarray(embeddings)
        n_lists = max(1, min(4096, int(math.sqrt(len(embeddings)))))
        params = ivf_flat.IndexParams(n_lists=n_lists)
        index = ivf_flat.build(params, device_embeddings)
        search_params = ivf_flat.SearchParams(n_probes=min(n_lists, 32))
        logger.info(
            "Built cuVS IVF-Flat index with %d vectors, %d lists, and %d probes",
            len(embeddings), n_lists, min(n_lists, 32)
        )
        return {
            "use_gpu": True,
            "index": index,
            "search_params": search_params,
            "size": len(embeddings)
        }

    logger.info("Falling back to CPU matrix multiplication search for %d vectors", len(embeddings))
    return {
        "use_gpu": False,
        "embeddings": embeddings,
        "size": len(embeddings)
    }


def batch_search(index,
                 query,
                 topk: int = 200,
                 batch_size: int = 64):
    all_distances, all_inxs = [], []
    use_gpu = index.get("use_gpu", False)

    if use_gpu:
        import cupy as cp  # type: ignore[import]
        from cuvs.neighbors import ivf_flat  # type: ignore[import]

        search_params = index["search_params"]
        gpu_index = index["index"]

    max_k = min(topk, index["size"])

    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = np.asarray(query[start_index:start_index + batch_size], dtype=np.float32)
        if use_gpu:
            device_query = cp.asarray(batch_query)
            batch_distances, batch_inxs = ivf_flat.search(search_params, gpu_index, device_query, max_k)
        else:
            raise NotImplementedError("CPU search is not implemented in this version.")
        #
        all_distances.extend(np.asarray(batch_distances).tolist())
        all_inxs.extend(np.asarray(batch_inxs).tolist())
    return all_distances, all_inxs


def get_corpus(candidate_pool):
    corpus = []
    for line in open(candidate_pool):
        line = json.loads(line.strip())
        corpus.append(line['text'])
    return corpus


def find_knn_neg(model, input_file, candidate_pool, output_file, sample_range, negative_number, filter_topk, batch_size, use_gpu):
    corpus = []
    queries = []
    train_data = []
    for line in open(input_file):
        line = json.loads(line.strip())
        if line['query'] in line['pos']:
            continue
        train_data.append(line)
        corpus.extend(line['pos'])
        if 'neg' in line:
            corpus.extend(line['neg'])
        queries.append(line['query'])

    if candidate_pool is not None and candidate_pool != "" and candidate_pool.lower() != "none":
        if not isinstance(candidate_pool, list):
            candidate_pool = get_corpus(candidate_pool)
        corpus = list(set(candidate_pool))
    else:
        corpus = list(set(corpus))

    print(f'inferencing embedding for corpus (number={len(corpus)})--------------')
    p_vecs = model.encode(corpus, batch_size=batch_size)
    p_vecs_dict = {p: p_v for p_v, p in zip(p_vecs, corpus)}

    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode_queries(queries, batch_size=batch_size)

    print('create index and search------------------')
    use_gpu = _should_use_gpu_for_search(use_gpu)
    index = create_index(p_vecs, use_gpu=use_gpu)
    all_distances, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1])
    assert len(all_inxs) == len(train_data)

    dump_data = []
    for i, data in enumerate(train_data):
        query = data['query']
        num_neg_to_add = negative_number - len(data['neg'])
        if num_neg_to_add <= 0:
            dump_data.append(data)
            continue
        
        q_v = q_vecs[i]
        p_v_list = np.array([p_vecs_dict[p] for p in data['pos']])
        pos_distances = np.linalg.norm(p_v_list - q_v, axis=1)  # euclidean distance
        pos_distance = np.min(pos_distances)
        all_p_distances = np.array(all_distances[i])

        if filter_topk is not None and pos_distance > all_p_distances[filter_topk]:
            # filter out the pos not in top-k(default 50)
            continue

        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        filtered_inx = []
        for inx in inxs:
            if inx == -1: break
            if corpus[inx] not in data['pos'] and corpus[inx] != query:
                filtered_inx.append(inx)

        if len(filtered_inx) > num_neg_to_add:
            filtered_inx = random.sample(filtered_inx, num_neg_to_add)
        # dedup
        mined_hard = set([corpus[inx] for inx in filtered_inx])    
        data['neg'].extend(list(mined_hard))
        
        dump_data.append(data)
    
    print(f"Data Keep: {len(dump_data)} / {len(train_data)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for data in dump_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    # clean
    del index


if __name__ == '__main__':
    parser = HfArgumentParser(HNMineArguments)
    args, = parser.parse_args_into_dataclasses()
    logger.info("Hard Negative Mining Parameters %s", args)

    sample_range = args.range_for_sampling.split('-')
    sample_range = [int(x) for x in sample_range]

    model = vLLMModel(model_name_or_path=args.model_name_or_path, 
                      pooling_method=args.pooling_method,
                      query_instruction_for_retrieval=args.query_instruction_for_retrieval, 
                      passages_instruction_for_retrieval=args.passages_instruction_for_retrieval,
                      use_fp16=False)

    find_knn_neg(model,
                 input_file=args.input_file,
                 candidate_pool=args.candidate_pool,
                 output_file=args.output_file,
                 sample_range=sample_range,
                 negative_number=args.negative_number,
                 filter_topk=args.filter_topk,
                 batch_size=args.batch_size,
                 use_gpu=args.use_gpu_for_searching)