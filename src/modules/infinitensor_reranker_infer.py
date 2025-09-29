import onnx
import numpy as np
import logging
from pyinfinitensor.onnx import OnnxStub, backend
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict, Any, Optional, Union
from tqdm import tqdm, trange
import time
from concurrent.futures import Future

from common.abs_reranker import AbstractRerank
from .request_queue import RequestQueue

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
        batch_size: int = 512,
        device: str = "cuda",
        normalize: bool = False,
        enable_queue: bool = True,
        queue_timeout: float = 0.1,
        use_cuda_graph: bool = True,
        **build_kwargs
    ) -> None:
        """Initialize InfinitensorRerankerInfer with ONNX model and tokenizer.

        Args:
            onnx_model_path: Path to the ONNX model file for InfiniTensor
            model_path: Path to the original model
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for inference
            device: Device to run inference on (cuda/cpu)
            normalize: Whether to normalize scores
            enable_queue: Whether to enable request queue for batching
            queue_timeout: Timeout in seconds for queue batching
            **build_kwargs: Additional arguments
        """
        self.onnx_model_path = onnx_model_path
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self.enable_queue = enable_queue
        self.queue_timeout = queue_timeout
        self.use_cuda_graph = use_cuda_graph
        # Load ONNX model
        self.onnx_model = onnx.load(onnx_model_path)

        # Initialize InfiniTensor backend
        if device == "cuda":
            self.backend = backend.cuda_runtime()
        else:
            self.backend = backend.cpu_runtime()

        # Create OnnxStub
        self.model = OnnxStub(self.onnx_model, self.backend)

        # Dry run to warm up the model
        if self.use_cuda_graph:
            self.model.run_with_cudagraph()
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

        # Initialize request queue if enabled
        if self.enable_queue:
            self.request_queue = RequestQueue(
                batch_size=self.batch_size,
                timeout_seconds=self.queue_timeout
            )
            self.request_queue.start_processing(self._process_batch_requests)
            self._request_counter = 0
        else:
            self.request_queue = None

    def _process_batch_requests(self, batch_tokenized_inputs: List[Dict]) -> List[float]:
        """Process a batch of pre-tokenized requests.

        Args:
            batch_tokenized_inputs: List of pre-tokenized inputs to process

        Returns:
            List of relevance scores
        """
        return self._compute_score_tokenized(batch_tokenized_inputs)

    def rerank_async(self, text_pair: Tuple[str, str]) -> Future:
        """Rerank a single text pair asynchronously using the queue.

        Args:
            text_pair: (query, document) text pair to rerank

        Returns:
            Future object that will contain the result
        """
        if not self.enable_queue or self.request_queue is None:
            raise RuntimeError("Queue is not enabled. Use rerank() method instead.")

        # Tokenize immediately while waiting for batch
        query, document = text_pair
        query_inputs = self.tokenizer(
            query,
            return_tensors=None,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        )['input_ids']

        document_inputs = self.tokenizer(
            document,
            return_tensors=None,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        )['input_ids']

        # Prepare tokenized input
        tokenized_input = self.tokenizer.prepare_for_model(
            query_inputs,
            document_inputs,
            truncation='only_second',
            max_length=self.max_length,
            padding="max_length",
        )

        self._request_counter += 1
        request_id = f"req_{self._request_counter}_{int(time.time() * 1000)}"
        return self.request_queue.submit_request(request_id, tokenized_input)

    def cleanup(self):
        """Clean up resources, especially the request queue."""
        if self.request_queue is not None:
            self.request_queue.stop_processing()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def rerank(self, texts: list[tuple[str, str]]) -> list[float]:
        """Rerank text pairs and return relevance scores.

        Args:
            texts: List of (query, document) text pairs to rerank

        Returns:
            List of relevance scores for each text pair
        """
        if self.enable_queue and self.request_queue is not None:
            # Use queue for batching
            futures = []
            for text_pair in texts:
                future = self.rerank_async(text_pair)
                futures.append(future)

            # Wait for all results
            scores = []
            for future in futures:
                try:
                    score = future.result(timeout=30.0)  # 30 second timeout
                    scores.append(score)
                except Exception as e:
                    raise RuntimeError(f"Error processing request: {e}")

            return scores
        else:
            # Process in batches for better performance (original behavior)
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

        all_scores = []
        for start_index in tqdm(range(0, len(all_inputs_sorted), batch_size), desc="Compute Scores",
                                disable=len(all_inputs_sorted) < batch_size):
            sentences_batch = all_inputs_sorted[start_index:start_index + batch_size]

            # Pad batch to full batch_size if needed
            padded_batch = sentences_batch[:]
            while len(padded_batch) < batch_size:
                # Add padding item (empty input)
                padded_batch.append({
                    'input_ids': [self.tokenizer.pad_token_id] * max_length,
                    'attention_mask': [0] * max_length
                })

            # Truncate if we have more than batch_size
            if len(padded_batch) > batch_size:
                padded_batch = padded_batch[:batch_size]

            # Prepare inputs for InfiniTensor - process as batch
            input_ids = np.array([item['input_ids'] for item in padded_batch])
            attention_mask = np.array([item['attention_mask'] for item in padded_batch])

            ort_inputs = {
                "input_ids": input_ids.astype(np.int64),
                "attention_mask": attention_mask.astype(np.int64),
            }

            # Copy inputs to InfiniTensor tensors
            for name, itensor in self.model.inputs.items():
                if name not in ort_inputs:
                    raise RuntimeError(f"Input {name} not found in tokenized inputs")
                itensor.copyin_numpy(ort_inputs[name])

            # Run model for the batch
            if self.use_cuda_graph:
                self.model.run_with_cudagraph()
            else:
                self.model.run()
            outputs = [output.copyout_numpy() for output in self.model.outputs.values()]
            cls_embeddings = outputs[0][:, 0, :]


            # Dense + tanh
            h = np.tanh(cls_embeddings @ self.W_dense.T + self.b_dense)

            # Out projection
            logits = h @ self.W_out.T + self.b_out  # [batch, 1]score

            # Only add scores for actual items (not padding)
            actual_batch_size = len(sentences_batch)
            for i in range(actual_batch_size):
                all_scores.append(float(logits[i][0]))


        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores

    def _compute_score_tokenized(self, tokenized_inputs: List[Dict]) -> List[float]:
        """Compute scores for pre-tokenized inputs.

        Args:
            tokenized_inputs: List of pre-tokenized inputs

        Returns:
            List of relevance scores
        """
        # Pad inputs to full batch size if needed
        padded_inputs = tokenized_inputs[:]
        while len(padded_inputs) < self.batch_size:
            # Add padding input
            padded_inputs.append({
                'input_ids': [self.tokenizer.pad_token_id] * self.max_length,
                'attention_mask': [0] * self.max_length
            })

        # Truncate if we have more than batch_size
        if len(padded_inputs) > self.batch_size:
            padded_inputs = padded_inputs[:self.batch_size]

        # Prepare inputs for InfiniTensor - process as batch
        input_ids = np.array([item['input_ids'] for item in padded_inputs])
        attention_mask = np.array([item['attention_mask'] for item in padded_inputs])

        ort_inputs = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        }

        # Copy inputs to InfiniTensor tensors
        for name, itensor in self.model.inputs.items():
            if name not in ort_inputs:
                raise RuntimeError(f"Input {name} not found in tokenized inputs")
            itensor.copyin_numpy(ort_inputs[name])

        # Run model for the batch
        self.model.run()
        outputs = [output.copyout_numpy() for output in self.model.outputs.values()]
        cls_embeddings = outputs[0][:, 0, :]


        # Dense + tanh
        h = np.tanh(cls_embeddings @ self.W_dense.T + self.b_dense)

        # Out projection
        logits = h @ self.W_out.T + self.b_out  # [batch, 1]score

        # Only return scores for actual items (not padding)
        actual_batch_size = len(tokenized_inputs)
        scores = []
        for i in range(actual_batch_size):
            scores.append(float(logits[i][0]))

        return scores
