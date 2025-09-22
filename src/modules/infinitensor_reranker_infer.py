import onnx
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Any, Optional, Union

from common.abs_reranker import AbstractRerank


class InfinitensorRerankerInfer(AbstractRerank):
    """
    InfiniTensor-based inference class for reranking models.

    This class provides reranking functionality using InfiniTensor ONNX models,
    combining the ONNX model loading patterns from InfinitensorInfer with
    the reranking interface from BgeRerankerInfer.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_length: int = 512,
        batch_size: int = 256,
        device: str = "cuda",
        **build_kwargs
    ) -> None:
        """Initialize InfinitensorRerankerInfer with ONNX model and tokenizer.

        Args:
            model_path: Path to the ONNX model file
            tokenizer_path: Path to the tokenizer
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for inference
            device: Device to run inference on (cuda/cpu)
            **build_kwargs: Additional arguments
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        # Load ONNX model
        self.onnx_model = onnx.load(model_path)

        # Initialize InfiniTensor backend
        if device == "cuda":
            self.backend = backend.cuda_runtime()
        else:
            self.backend = backend.cpu_runtime()

        # Create OnnxStub
        self.model = OnnxStub(self.onnx_model, self.backend)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def rerank(self, texts: list[tuple[str, str]]) -> list[float]:
        """Rerank text pairs and return relevance scores.

        Args:
            texts: List of (query, document) text pairs to rerank

        Returns:
            List of relevance scores for each text pair
        """
        # Process in batches for better performance
        results = []

        for start_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[start_idx:start_idx + self.batch_size]
            batch_results = self._rerank_batch(batch_texts)
            results.extend(batch_results)

        return results

    def _rerank_batch(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Rerank a batch of text pairs using InfiniTensor ONNX model.

        Args:
            text_pairs: List of (query, document) text pairs

        Returns:
            List of relevance scores for the batch
        """
        batch_results = []

        for query, document in text_pairs:
            # Concatenate query and document for reranking
            # Format: [CLS] query [SEP] document [SEP]
            combined_text = f"{query} [SEP] {document}"

            # Tokenize input text with max_length padding (InfiniTensor requirement)
            encoded = self.tokenizer(
                combined_text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
            )

            # Prepare inputs for ONNX model
            ort_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }

            # Copy inputs to InfiniTensor tensors
            for name, itensor in self.model.inputs.items():
                if name not in ort_inputs:
                    raise RuntimeError(f"Input {name} not found in tokenized inputs")
                itensor.copyin_numpy(ort_inputs[name])

            # Run inference
            self.model.run()

            # Extract outputs
            outputs = [output.copyout_numpy() for output in self.model.outputs.values()]

            if len(outputs) == 0:
                raise RuntimeError("No outputs from model")

            # Extract relevance score (typically the first output)
            # Apply sigmoid if needed to get probability-like scores
            score = outputs[0].item()

            # Apply sigmoid activation if the model outputs logits
            # This is common for reranking models
            if score < -10 or score > 10:  # Likely logits, apply sigmoid
                score = 1.0 / (1.0 + np.exp(-score))

            batch_results.append(float(score))

        return batch_results
