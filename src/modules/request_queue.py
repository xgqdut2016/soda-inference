import queue
import threading
import time
from dataclasses import dataclass
from concurrent.futures import Future
from typing import List, Any, Callable


@dataclass
class RequestItem:
    """Represents a single request in the queue."""
    request_id: str
    data: Any
    future: Future
    timestamp: float


class RequestQueue:
    """A reusable queue system for batching requests with timeout and size conditions."""

    def __init__(self, batch_size: int, timeout_seconds: float = 0.1):
        """Initialize the request queue.

        Args:
            batch_size: Maximum number of requests to batch together
            timeout_seconds: Maximum time to wait before processing a batch
        """
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.queue = queue.Queue()
        self.processing = False
        self.thread = None
        self.stop_event = threading.Event()

    def start_processing(self, process_func: Callable):
        """Start the background processing thread.

        Args:
            process_func: Function to process batches of requests
        """
        if self.thread is None or not self.thread.is_alive():
            self.processing = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._process_requests, args=(process_func,))
            self.thread.daemon = True
            self.thread.start()

    def stop_processing(self):
        """Stop the background processing thread."""
        self.processing = False
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def submit_request(self, request_id: str, data: Any) -> Future:
        """Submit a request to the queue.

        Args:
            request_id: Unique identifier for the request
            data: Request data

        Returns:
            Future object that will contain the result
        """
        future = Future()
        request_item = RequestItem(
            request_id=request_id,
            data=data,
            future=future,
            timestamp=time.time()
        )
        self.queue.put(request_item)
        return future

    def _process_requests(self, process_func: Callable):
        """Background thread that processes requests in batches."""
        batch = []
        last_batch_time = time.time()

        while self.processing and not self.stop_event.is_set():
            try:
                # Try to get a request with timeout
                try:
                    request_item = self.queue.get(timeout=0.1)
                    batch.append(request_item)
                except queue.Empty:
                    # No new requests, check if we should process current batch
                    if batch and (time.time() - last_batch_time) >= self.timeout_seconds:
                        self._process_batch(batch, process_func)
                        batch = []
                        last_batch_time = time.time()
                    continue

                # Check if batch is full
                if len(batch) >= self.batch_size:
                    self._process_batch(batch, process_func)
                    batch = []
                    last_batch_time = time.time()

            except Exception as e:
                # Handle any errors in processing
                for item in batch:
                    item.future.set_exception(e)
                batch = []

        # Process any remaining requests
        if batch:
            self._process_batch(batch, process_func)

    def _process_batch(self, batch: List[RequestItem], process_func: Callable):
        """Process a batch of requests.

        Args:
            batch: List of request items to process
            process_func: Function to process the batch
        """
        try:
            # Extract data from batch
            batch_data = [item.data for item in batch]

            # Process the batch
            results = process_func(batch_data)

            # Set results for each future
            for i, item in enumerate(batch):
                if i < len(results):
                    item.future.set_result(results[i])
                else:
                    item.future.set_exception(IndexError("Result index out of range"))

        except Exception as e:
            # Set exception for all futures in the batch
            for item in batch:
                item.future.set_exception(e)
