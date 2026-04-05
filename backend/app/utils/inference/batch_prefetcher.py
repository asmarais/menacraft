from typing import Iterator, Tuple, List, Dict, Any, Optional
import numpy as np
from queue import Queue
from threading import Thread
from logging import getLogger


class BatchPrefetcher:
    """Prefetch and batch frames in background thread for optimal GPU utilization.

    While the GPU processes the current batch, the CPU extracts and prepares
    the next batches in parallel, minimizing idle time.
    """

    def __init__(self,
        frame_iterator: Iterator[Tuple[np.ndarray, Dict[str, Any]]],
        batch_size: int,
        prefetch_size: int =2
    ):
        """Initialize the batch prefetcher.
        Parameters:
            frame_iterator: iterator yielding (frame, metadata) tuples
            batch_size (int): number of frames per batch
            prefetch_size (int): number of batches to prefetch ahead (default: 2)
        """
        self.frame_iterator = frame_iterator
        """Iterator yielding (frame, metadata) tuples."""
        self.batch_size = batch_size
        """Number of frames per batch to prefetch."""
        self.queue: Queue = Queue(maxsize=prefetch_size)
        """Queue to hold prefetched batches."""
        self.prefetch_thread: Optional[Thread] = None
        """Background thread for prefetching."""
        self.stop_flag = False
        """Flag to signal stopping of prefetching."""
        self.logger = getLogger("/".join(__file__.split("/")[-2:]))
        """Logger for debugging."""

    def _prefetch_worker(self) -> None:
        """Background worker that prefetches batches."""
        try:
            batch_frames: List[np.ndarray] = []
            batch_metadata: List[Dict[str, Any]] = []

            for frame, metadata in self.frame_iterator:
                if self.stop_flag:
                    break

                batch_frames.append(frame)
                batch_metadata.append(metadata)

                if len(batch_frames) == self.batch_size:
                    #put batch in queue (blocks if queue is full)
                    self.queue.put((batch_frames, batch_metadata))
                    batch_frames = []
                    batch_metadata = []

            #put remaining frames as final batch
            if batch_frames and not self.stop_flag:
                self.queue.put((batch_frames, batch_metadata))

            #signal end of iteration
            self.queue.put(None)

        except Exception as e:
            self.logger.error(f"Prefetch worker failed: {e}")
            self.queue.put(None)

    def start(self) -> None:
        """Start prefetching in background thread."""
        self.prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        self.logger.debug("Prefetch worker started")

    def __iter__(self):
        """Iterator interface."""
        return self

    def __next__(self) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Get next prefetched batch.
        Returns:
            Tuple of (frames, metadata) for the batch
        Raises:
            StopIteration: when no more batches available
        """
        batch = self.queue.get()

        if batch is None:
            raise StopIteration

        return batch

    def stop(self) -> None:
        """Stop prefetching and cleanup."""
        self.stop_flag = True
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
