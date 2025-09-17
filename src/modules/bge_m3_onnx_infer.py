from collections import defaultdict
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from transformers import AutoTokenizer
from common.abs_embedding import AbstractEmbed

class BgeM3OnnxInfer(AbstractEmbed):
    def __init__(
            self,
            tokenizer_path: str,
            model_onnx_path: str,
            providers=['CUDAExecutionProvider'],
            provider_options=None,
            padding='longest',
            max_length=8192,
            batch_size=12,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.ort_session = ort.InferenceSession(model_onnx_path, providers=providers, provider_options=provider_options)
        self.unused_tokens = set([
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ])
        self.padding = padding
        self.max_length = max_length
        self.batch_size = batch_size

    def _process_token_weights(self, token_weights: np.ndarray, input_ids: list):
        # conver to dict
        result = dict()
        # token_weights = np.ceil(token_weights * 100)
        for w, idx in zip(token_weights, input_ids):
            if idx not in self.unused_tokens and w > 0:
                idx = str(idx)
                if w > result.get(idx, 0):
                    result[idx] = float(w)
        return result

    def _process_colbert_vecs(self, colbert_vecs: np.ndarray, attention_mask: list):
        # delte the vectors of padding tokens
        tokens_num = np.sum(attention_mask)
        return colbert_vecs[:tokens_num - 1]  # we don't use the embedding of cls, so select tokens_num-1

    def encode(self, texts: list[str]) -> list[tuple[list[float], dict[str, float]]]:
        all_res = []
        for start_index in tqdm(range(0, len(texts), self.batch_size), desc="Inference Embeddings",
                                disable=len(texts) < self.batch_size):
            texts_batch = texts[start_index:start_index + self.batch_size]
            res_batch = self.encode_batch(texts_batch)
            all_res.extend(res_batch)
        return all_res

    def encode_batch(self, texts: list[str]) -> list[tuple[list[float], dict[str, float]]]:
        inputs = self.tokenizer(texts, padding=self.padding, return_tensors='np', max_length=self.max_length)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        output_tensors = self.ort_session.run(None, dict(
            input_ids = ort.OrtValue.ortvalue_from_numpy(input_ids),
            attention_mask = ort.OrtValue.ortvalue_from_numpy(attention_mask),
        ))

        denses = output_tensors[0].tolist()
        sparses = list(map(self._process_token_weights, output_tensors[1], input_ids.tolist()))
        assert len(denses) == len(texts)
        assert len(sparses) == len(texts)
        return list(zip(denses, sparses))

    def can_encode(self, return_dense: bool, return_sparse: bool) -> str | None:
        return None

