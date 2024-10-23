import concurrent.futures
import threading
import queue

class CappedFuture(concurrent.futures.Future):
    def __init__(self, semaphore):
        super().__init__()
        self._semaphore = semaphore
        self._result_retrieved = False
        self._underlying_future = None
        self._condition = threading.Condition()

    def set_underlying_future(self, underlying_future):
        with self._condition:
            self._underlying_future = underlying_future
            # Transfer the result when the underlying future completes
            underlying_future.add_done_callback(self._transfer_result)

    def _transfer_result(self, underlying_future):
        if underlying_future.cancelled():
            self.set_cancelled()
        elif underlying_future.exception() is not None:
            self.set_exception(underlying_future.exception())
        else:
            try:
                result = underlying_future.result()
                self.set_result(result)
            except Exception as e:
                self.set_exception(e)

    def result(self, timeout=None):
        res = super().result(timeout)
        self._release_semaphore()
        return res

    def exception(self, timeout=None):
        exc = super().exception(timeout)
        self._release_semaphore()
        return exc

    def _release_semaphore(self):
        if not self._result_retrieved:
            self._result_retrieved = True
            self._semaphore.release()

    def cancel(self):
        with self._condition:
            if self._underlying_future is not None:
                cancelled = self._underlying_future.cancel()
                if cancelled:
                    super().cancel()
                return cancelled
            else:
                # Task has not been submitted yet; cancel directly
                return super().cancel()

    def cancelled(self):
        return super().cancelled()

    def running(self):
        with self._condition:
            if self._underlying_future is not None:
                return self._underlying_future.running()
            else:
                return False

    def done(self):
        return super().done()

class CappedProcessPoolExecutor(concurrent.futures.Executor):
    def __init__(self, max_unprocessed=100, max_workers=None):
        self._max_unprocessed = max_unprocessed
        self._semaphore = threading.BoundedSemaphore(max_unprocessed)
        self._task_queue = queue.Queue()
        self._shutdown = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        self._worker_thread = threading.Thread(target=self._worker)
        self._worker_thread.daemon = True
        self._worker_thread.start()

    def submit(self, fn, *args, **kwargs):
        if self._shutdown.is_set():
            raise RuntimeError('Cannot submit new tasks after shutdown')
        # Create a CappedFuture to return to the user
        user_future = CappedFuture(self._semaphore)
        # Put the task in the queue
        self._task_queue.put((user_future, fn, args, kwargs))
        return user_future

    def _worker(self):
        while True:
            if self._shutdown.is_set() and self._task_queue.empty():
                break
            try:
                user_future, fn, args, kwargs = self._task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._semaphore.acquire()
            if user_future.cancelled():
                self._semaphore.release()
                continue
            # Submit the task to the underlying executor
            try:
                underlying_future = self._executor.submit(fn, *args, **kwargs)
                user_future.set_underlying_future(underlying_future)
            except Exception as e:
                user_future.set_exception(e)
                self._semaphore.release()
                continue

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            self._shutdown.set()
            self._worker_thread.join()
            self._executor.shutdown(wait=wait)
