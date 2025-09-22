from common.abs_embedding import AbstractEmbed
from common.abs_reranker import AbstractRerank
from common.abs_rewrite import AbstractReWrite
from modules.bce_reranker_infer import BceRerankerInfer
from modules.bge_m3_infer import BgeM3Infer
from modules.bge_m3_onnx_infer import BgeM3OnnxInfer
from modules.bge_reranker_infer import BgeRerankerInfer
from modules.infinitensor_infer import InfinitensorInfer
from modules.infinitensor_reranker_infer import InfinitensorRerankerInfer
from modules.fastembed_dense import FastembedDense
from modules.fastembed_sparse import FastembedSparse
from modules.optimum_onnx_reranker import OptimumOnnxReranker
from modules.mt5_rewrite_infer import MT5RewriteInfer

MODEL_DICT_EMBED: dict[str, type[AbstractEmbed]] = {}
MODEL_DICT_RERANK: dict[str, type[AbstractRerank]] = {}
MODEL_DICT_REWRITE: dict[str, type[AbstractReWrite]] = {}

for model_cls in (
        BgeM3OnnxInfer,
        BgeM3Infer,
        InfinitensorInfer,
        FastembedDense,
        FastembedSparse,
    ):
    MODEL_DICT_EMBED[model_cls.__name__] = model_cls

for model_cls in (
        BceRerankerInfer,
        BgeRerankerInfer,
        InfinitensorRerankerInfer,
        OptimumOnnxReranker,
    ):
    MODEL_DICT_RERANK[model_cls.__name__] = model_cls


for model_cls in (
        MT5RewriteInfer,
    ):
    print(model_cls.__name__)
    MODEL_DICT_REWRITE[model_cls.__name__] = model_cls
