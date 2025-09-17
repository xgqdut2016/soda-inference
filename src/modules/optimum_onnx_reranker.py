import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from common.abs_reranker import AbstractRerank
from optimum.onnxruntime import ORTModelForSequenceClassification

# 如何对 rerank 导出 onnx
# optimum-cli export onnx -m /home/zhuchunyang/tools/bge-reranker-v2-m3 --optimize O4 rerank-bge_v2_m3-onnx_cuda_o4 --device cuda --task text-classification --dtype fp32
# optimum-cli export onnx -m /home/zhuchunyang/tools/bce-reranker-base_v1 --optimize O4 rerank-bce_v1-onnx_cuda_o4 --device cuda --task text-classification --dtype fp32

# 如何使用
# SERVICE_PORT=10996 MODEL_TYPE=rerank MODEL_CLS=OptimumOnnxReranker MODEL_KWARGS='{"model_path":"/home/zhuchunyang/tools/rerank-bge_v2_m3-onnx_cuda_o4"}' python start_server.py
# SERVICE_PORT=10999 MODEL_TYPE=rerank MODEL_CLS=OptimumOnnxReranker MODEL_KWARGS='{"model_path":"/home/zhuchunyang/tools/rerank-bce_v1-onnx_cuda_o4","score_sigmoid":true}' python start_server.py
# 注：bce 模型需设置 score_sigmoid=True

class OptimumOnnxReranker(AbstractRerank):
    def __init__(
            self,
            model_path: str,
            provider='CUDAExecutionProvider',
            provider_options=None,
            padding=True,
            truncation=True,
            score_sigmoid=False,
            max_length=512,
            batch_size=12,
            device: str | None='cuda',
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
        )
        self.model: ORTModelForSequenceClassification = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            provider=provider,
            provider_options=provider_options,
        )
        if device:
            self.model.to(device)
        self.truncation = truncation
        self.padding = padding
        self.score_sigmoid = score_sigmoid
        self.max_length = max_length
        self.batch_size = batch_size

    def rerank(self, texts: list[tuple[str, str]]) -> list[float]:
        all_scores = []
        for start_index in range(0, len(texts), self.batch_size):
            texts_batch = texts[start_index:start_index + self.batch_size]
            scores = self.rerank_batch(texts_batch)
            all_scores.extend(scores)
        return all_scores

    def rerank_batch(self, texts: list[tuple[str, str]]) -> list[float]:
        inputs = self.tokenizer(texts, padding=self.padding, truncation=self.truncation, return_tensors='pt', max_length=self.max_length)
        output = self.model.forward(**inputs, return_dict=True)
        scores = output.logits.view(-1, ).float()
        if self.score_sigmoid:
            scores = torch.sigmoid(scores)
        return scores.tolist()
