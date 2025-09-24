import onnx
import numpy as np
import logging
from pyinfinitensor.onnx import OnnxStub, backend
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict, Any, Optional, Union
from tqdm import tqdm, trange

from common.abs_reranker import AbstractRerank

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))


class InfinitensorRerankerInfer(AbstractRerank):
    """
    InfiniTensor-based inference class for reranking models.

    This class provides reranking functionality using InfiniTensor ONNX models,
    combining the ONNX model loading patterns from InfinitensorInfer with
    the reranking interface from BgeRerankerInfer.
    """

    def __init__(
        self,
        onnx_model_path: str,
        model_path: str,
        max_length: int = 512,
        batch_size: int = 256,
        device: str = "cuda",
        normalize: bool = False,
        **build_kwargs
    ) -> None:
        """Initialize InfinitensorRerankerInfer with ONNX model and tokenizer.

        Args:
            onnx_model_path: Path to the ONNX model file for InfiniTensor
            model_path: Path to the original model
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for inference
            device: Device to run inference on (cuda/cpu)
            **build_kwargs: Additional arguments
        """
        self.onnx_model_path = onnx_model_path
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize

        # Load ONNX model
        self.onnx_model = onnx.load(onnx_model_path)

        # Initialize InfiniTensor backend
        if device == "cuda":
            self.backend = backend.cuda_runtime()
        else:
            self.backend = backend.cpu_runtime()

        # Create OnnxStub
        self.model = OnnxStub(self.onnx_model, self.backend)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.original_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # Load weights from reference model
        dense = self.original_model.classifier.dense       # Linear(1024, 1024)
        out_proj = self.original_model.classifier.out_proj # Linear(1024, 1)

        self.W_dense = dense.weight.detach().cpu().numpy()  # shape [1024, 1024]
        self.b_dense = dense.bias.detach().cpu().numpy()    # shape [1024]

        self.W_out = out_proj.weight.detach().cpu().numpy() # shape [1, 1024]
        self.b_out = out_proj.bias.detach().cpu().numpy()   # shape [1]



    def rerank(self, texts: list[tuple[str, str]]) -> list[float]:
        """Rerank text pairs and return relevance scores.

        Args:
            texts: List of (query, document) text pairs to rerank

        Returns:
            List of relevance scores for each text pair
        """
        # Process in batches for better performance
        scores = self._compute_scores(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        if isinstance(scores, float):
            scores = [scores]
        return scores

    def _compute_scores(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        **kwargs
    ) -> List[float]:
        return self._compute_score_once(
            sentence_pairs,
        )

    def _compute_score_once(
        self, sentence_pairs: List[Tuple[str, str]],
        batch_size: Optional[int] = None,
        query_max_length: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        **kwargs: Any
    ) -> List[float]:
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.max_length
        if normalize is None: normalize = self.normalize

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        all_inputs = []
        for start_index in trange(0, len(sentence_pairs), batch_size, desc="pre tokenize",
                                  disable=len(sentence_pairs) < batch_size):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            queries = [s[0] for s in sentences_batch]
            passages = [s[1] for s in sentences_batch]
            queries_inputs_batch = self.tokenizer(
                queries,
                return_tensors=None,
                add_special_tokens=False,
                max_length=query_max_length,
                truncation=True,
                **kwargs
            )['input_ids']
            passages_inputs_batch = self.tokenizer(
                 passages,
                 return_tensors=None,
                 add_special_tokens=False,
                 max_length=max_length,
                 truncation=True,
                 **kwargs
             )['input_ids']
            for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
                logger.debug(f"q_inp: {q_inp}, d_inp: {d_inp}")
                item = self.tokenizer.prepare_for_model(
                     q_inp,
                     d_inp,
                     truncation='only_second',
                     max_length=max_length,
                     padding="max_length",  # No padding here, like FlagReranker
                )
                all_inputs.append(item)
        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]
        logger.debug(f"all_inputs_sorted: {all_inputs_sorted}")

        all_scores = []
        for start_index in tqdm(range(0, len(all_inputs_sorted), batch_size), desc="Compute Scores",
                                disable=len(all_inputs_sorted) < batch_size):
            sentences_batch = all_inputs_sorted[start_index:start_index + batch_size]

            # Process each sentence pair individually
            for sentence_item in sentences_batch:
                # Use tokenizer.pad() for single item
                inputs = self.tokenizer.pad(
                    [sentence_item],  # Wrap single item in list
                    padding=True,
                    return_tensors='np',  # Return numpy arrays for InfiniTensor
                    **kwargs
                )

                # Prepare inputs for InfiniTensor
                ort_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                }
                logger.debug(f"ort_inputs shapes: input_ids={ort_inputs['input_ids'].shape}, attention_mask={ort_inputs['attention_mask'].shape}")

                # Copy inputs to InfiniTensor tensors
                for name, itensor in self.model.inputs.items():
                    if name not in ort_inputs:
                        raise RuntimeError(f"Input {name} not found in tokenized inputs")
                    itensor.copyin_numpy(ort_inputs[name])

                # Run model for this single sentence pair
                self.model.run()
                outputs = [output.copyout_numpy() for output in self.model.outputs.values()]
                cls_embeddings = outputs[0][:, 0, :]

                logger.debug(f"outputs: {outputs}, outputs[0].shape: {outputs[0].shape}")

                # Dense + tanh
                h = np.tanh(cls_embeddings @ self.W_dense.T + self.b_dense)

                # Out projection
                logits = h @ self.W_out.T + self.b_out  # [batch, 1]score
                logger.debug(f"logits: {logits}, logits.shape: {logits.shape}")
                all_scores.append(float(logits[0][0]))


        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores
