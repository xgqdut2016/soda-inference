import os
import json


PROMETHEUS_SERVICE_PORT = int(os.getenv('PROMETHEUS_SERVICE_PORT', 0))
SERVICE_PORT = int(os.getenv('SERVICE_PORT', 9991))

REST_TIMEOUT = float(os.getenv('REST_TIMEOUT', 15.0))

MODEL_TYPE = os.getenv('MODEL_TYPE', '')
MODEL_NAME = os.getenv('MODEL_NAME', 'default')
MODEL_CLS = os.getenv('MODEL_CLS', '')
MODEL_KWARGS = json.loads(os.getenv('MODEL_KWARGS', '{}'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 12))
QUEUE_SIZE = int(os.getenv('QUEUE_SIZE', 1024))

## --- Bge m3
# SERVICE_PORT=10991 MODEL_TYPE=embed MODEL_CLS=BgeM3Infer MODEL_KWARGS='{"model_path": "/home/zenghua/repos/soda-inference/bge-m3","use_fp16":false}' python src/start_server.py
# SERVICE_PORT=10991 MODEL_TYPE=embed MODEL_CLS=InfinitensorInfer MODEL_KWARGS='{"model_path": "/home/zenghua/repos/soda-inference/bge-m3"}' python src/start_server.py
# SERVICE_PORT=10992 MODEL_TYPE=embed MODEL_CLS=InfinitensorInfer MODEL_KWARGS='{"model_path": "/home/zenghua/BGE/bge_sim_512.onnx","tokenizer_path": "/home/zenghua/repos/soda-inference/bge-m3"}' python src/start_server.py
# SERVICE_PORT=10992 MODEL_TYPE=embed MODEL_CLS=BgeM3Infer MODEL_KWARGS='{"model_path": "/home/zhuchunyang/tools/bge-m3","use_fp16":true}' python start_server.py
## --- Bge m3

## --- Bge m3 onnx
# SERVICE_PORT=10993 MODEL_TYPE=embed MODEL_CLS=BgeM3OnnxInfer MODEL_KWARGS='{"tokenizer_path":"/home/zhuchunyang/tools/bge_m3_onnx/tokenizer","model_onnx_path":"/home/zhuchunyang/tools/bge_m3_onnx/onnx_model/bge_m3_fp16_dense_sparse_optimized.onnx"}' python start_server.py
# SERVICE_PORT=10993 MODEL_TYPE=embed MODEL_CLS=BgeM3OnnxInfer MODEL_KWARGS='{"tokenizer_path":"/Users/qurakchin/Works/optimized_for_infer/bge_m3_onnx/tokenizer","model_onnx_path":"/home/zhuchunyang/tools/bge_m3_onnx/onnx_model/bge_m3_fp16_dense_sparse_optimized.onnx"}' python start_server.py
## --- Bge m3 onnx

## --- Bge rerank
# SERVICE_PORT=10993 MODEL_TYPE=rerank MODEL_CLS=BgeRerankerInfer MODEL_KWARGS='{"model_path":"/home/zenghua/repos/soda-inference/bge-reranker-v2-m3","use_fp16":false}' python src/start_server.py
# SERVICE_PORT=10994 MODEL_TYPE=rerank MODEL_CLS=InfinitensorRerankerInfer MODEL_KWARGS='{"onnx_model_path":"/home/zenghua/BGE-reranker-512/bge_reranker_O1_sim_512.onnx","model_path":"/home/zenghua/repos/soda-inference/bge-reranker-v2-m3"}' python src/start_server.py
# SERVICE_PORT=10994 MODEL_TYPE=rerank MODEL_CLS=InfinitensorRerankerInfer MODEL_KWARGS='{"onnx_model_path":"/home/zenghua/BGE-reranker-512-logits/bge_reranker.sim.onnx","model_path":"/home/zenghua/repos/soda-inference/bge-reranker-v2-m3"}' python src/start_server.py
# SERVICE_PORT=10995 MODEL_TYPE=rerank MODEL_CLS=BgeRerankerInfer MODEL_KWARGS='{"model_path":"/home/zhuchunyang/tools/bge-reranker-v2-m3","use_fp16":true}' python start_server.py
## --- Bge rerank

## --- Bge reranker onnx
# SERVICE_PORT=10996 MODEL_TYPE=rerank MODEL_CLS=OptimumOnnxReranker MODEL_KWARGS='{"model_path":"/home/zhuchunyang/tools/rerank-bge_v2_m3-onnx_cuda_o4"}' python start_server.py
## --- Bge reranker onnx

## --- Bce rerank
# SERVICE_PORT=10997 MODEL_TYPE=rerank MODEL_CLS=BceRerankerInfer MODEL_KWARGS='{"model_path":"/home/zhuchunyang/tools/bce-reranker-base_v1","use_fp16":false}' python start_server.py
# SERVICE_PORT=10998 MODEL_TYPE=rerank MODEL_CLS=BceRerankerInfer MODEL_KWARGS='{"model_path":"/home/zhuchunyang/tools/bce-reranker-base_v1","use_fp16":true}' python start_server.py
## --- Bce rerank

## --- Bce reranker onnx
# SERVICE_PORT=10999 MODEL_TYPE=rerank MODEL_CLS=OptimumOnnxReranker MODEL_KWARGS='{"model_path":"/home/zhuchunyang/tools/rerank-bce_v1-onnx_cuda_o4","score_sigmoid":true}' python start_server.py
## --- Bce reranker onnx

## --- fastembed -- dense -- BAAI/bge-small-en-v1.5
# SERVICE_PORT=11991 MODEL_TYPE=embed MODEL_CLS=FastembedDense MODEL_KWARGS='{"model_name":"BAAI/bge-small-en-v1.5","model_path":"/Users/qurakchin/Works/optimized_for_infer/bge-small-en-v1.5","providers":["CoreMLExecutionProvider"]}' python start_server.py
## --- fastembed -- dense -- BAAI/bge-small-en-v1.5

## --- fastembed -- sparse -- prithivida/Splade_PP_en_v1
# SERVICE_PORT=11991 MODEL_TYPE=embed MODEL_CLS=FastembedSparse MODEL_KWARGS='{"model_name":"prithivida/Splade_PP_en_v1","model_path":"/Users/qurakchin/Works/optimized_for_infer/Splade_PP_en_v1","providers":["CoreMLExecutionProvider"]}' python start_server.py
## --- fastembed -- sparse -- prithivida/Splade_PP_en_v1
