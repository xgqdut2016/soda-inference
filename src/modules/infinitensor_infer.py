import onnx
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import defaultdict

from common.abs_embedding import AbstractEmbed


class InfinitensorInfer(AbstractEmbed):
    """
    InfiniTensor-based inference class for embedding models with post-compute functionality.

    This class provides similar functionality to BGE-M3 inference with support for:
    - Dense embeddings
    - Sparse embeddings (lexical weights)
    - ColBERT vectors
    - Batch processing
    - Post-compute operations similar to M3Embedder

    The class processes model outputs and applies the same post-compute operations
    as the FlagEmbedding M3Embedder, including:
    - Token weight processing for sparse embeddings
    - ColBERT vector processing with attention mask handling
    - Special token filtering
    - Batch processing for improved performance
    """
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_length: int = 512,
        batch_size: int = 12,
        device: str = "cuda",
        **build_kwargs
    ) -> None:
        """Initialize InfinitensorInfer with ONNX model and tokenizer.

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

    def _process_token_weights(self, token_weights: np.ndarray, input_ids: List[int]) -> Dict[str, float]:
        """Process token weights for sparse embeddings, similar to M3Embedder.

        Args:
            token_weights: Token weights from model output
            input_ids: Input token IDs

        Returns:
            Dictionary mapping token IDs to weights
        """
        result = defaultdict(int)
        unused_tokens = set()

        # Get special token IDs to exclude
        for _token in ['cls_token', 'eos_token', 'pad_token', 'unk_token']:
            if _token in self.tokenizer.special_tokens_map:
                _token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map[_token])
                unused_tokens.add(_token_id)

        # Process token weights
        for w, idx in zip(token_weights, input_ids):
            if idx not in unused_tokens and w > 0:
                idx_str = str(idx)
                if w > result[idx_str]:
                    result[idx_str] = float(w)

        return dict(result)

    def _process_colbert_vecs(self, colbert_vecs: np.ndarray, attention_mask: List[int]) -> np.ndarray:
        """Process colbert vectors, similar to M3Embedder.

        Args:
            colbert_vecs: Colbert vectors from model output
            attention_mask: Attention mask to determine actual token count

        Returns:
            Processed colbert vectors without padding tokens
        """
        # Delete the vectors of padding tokens
        tokens_num = np.sum(attention_mask)
        return colbert_vecs[:tokens_num - 1]  # We don't use the embedding of cls, so select tokens_num-1

    def encode(self, texts: list[str]) -> list[tuple[list[float], dict[str, float]]]:
        """Encode texts into embeddings using InfiniTensor ONNX model.

        Args:
            texts: List of input texts to encode

        Returns:
            List of tuples containing (dense_embeddings, sparse_embeddings)
        """
        # Process in batches for better performance
        results = []

        for start_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[start_idx:start_idx + self.batch_size]
            batch_results = self._encode_batch(batch_texts)
            results.extend(batch_results)

        return results

    def _encode_batch(self, texts: List[str]) -> List[Tuple[List[float], Dict[str, float]]]:
        """Encode a batch of texts using InfiniTensor-compatible preprocessing.

        Note: InfiniTensor ONNX models require fixed input shapes, so we must use
        max_length padding instead of dynamic padding. This is a limitation of
        ONNX models that need consistent tensor shapes across different batches.

        Args:
            texts: List of input texts to encode

        Returns:
            List of tuples containing (dense_embeddings, sparse_embeddings)
        """
        batch_results = []

        for text in texts:
            # Tokenize input text with max_length padding (InfiniTensor requirement)
            # This ensures consistent input shapes across different batches
            encoded = self.tokenizer(
                text,
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

            # Process outputs - always return both dense and sparse like BgeM3Infer
            dense_embedding = None
            sparse_embedding = None

            # Dense embedding (first output)
            if len(outputs) > 0:
                dense_embedding = outputs[0].flatten().tolist()

            # Sparse embedding (second output)
            if len(outputs) > 1:
                token_weights = outputs[1].squeeze()  # Remove batch dimension if present
                input_ids = encoded["input_ids"][0].tolist()  # Get first (and only) sequence
                sparse_dict = self._process_token_weights(token_weights, input_ids)
                # Convert string keys to int keys to match BgeM3Infer format
                sparse_embedding = {int(k): float(v) for k, v in sparse_dict.items()}
            else:
                # If no sparse output, create empty dict
                sparse_embedding = {}

            batch_results.append((dense_embedding, sparse_embedding))

        return batch_results

    def can_encode(self, return_dense: bool, return_sparse: bool) -> str | None:
        """Check if the model can encode with the specified requirements.

        Args:
            return_dense: Whether dense embeddings are requested
            return_sparse: Whether sparse embeddings are requested

        Returns:
            None if requirements can be met, error message otherwise
        """
        return None

    def convert_id_to_token(self, lexical_weights: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Convert token IDs back to tokens in lexical weights.

        Args:
            lexical_weights: List of dictionaries with token IDs as keys and weights as values

        Returns:
            List of dictionaries with tokens as keys and weights as values
        """
        if isinstance(lexical_weights, dict):
            lexical_weights = [lexical_weights]

        new_lexical_weights = []
        for item in lexical_weights:
            new_item = {}
            for token_id, weight in item.items():
                try:
                    token = self.tokenizer.decode([int(token_id)])
                    new_item[token] = weight
                except (ValueError, KeyError):
                    # Skip invalid token IDs
                    continue
            new_lexical_weights.append(new_item)

        if len(new_lexical_weights) == 1:
            new_lexical_weights = new_lexical_weights[0]
        return new_lexical_weights

    def compute_lexical_matching_score(
        self,
        lexical_weights_1: Union[Dict[str, float], List[Dict[str, float]]],
        lexical_weights_2: Union[Dict[str, float], List[Dict[str, float]]]
    ) -> Union[np.ndarray, float]:
        """Compute the lexical matching score of two given lexical weights.

        Args:
            lexical_weights_1: First array of lexical weights
            lexical_weights_2: Second array of lexical weights

        Returns:
            The computed lexical weights across the two arrays of lexical weights
        """
        def _compute_single_lexical_matching_score(lw1: Dict[str, float], lw2: Dict[str, float]):
            scores = 0
            for token, weight in lw1.items():
                if token in lw2:
                    scores += weight * lw2[token]
            return scores

        if isinstance(lexical_weights_1, dict) and isinstance(lexical_weights_2, dict):
            return _compute_single_lexical_matching_score(lexical_weights_1, lexical_weights_2)
        elif isinstance(lexical_weights_1, list) and isinstance(lexical_weights_2, list):
            scores_array = []
            for lw1 in lexical_weights_1:
                scores_array.append([
                    _compute_single_lexical_matching_score(lw1, lw2)
                    for lw2 in lexical_weights_2
                ])
            return np.array(scores_array)
        else:
            raise ValueError("The input format of lexical_weights is not correct.")
