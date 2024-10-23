import unittest
import time
import concurrent.futures
from concurrent.futures import TimeoutError

# Assuming the CappedProcessPoolExecutor code is in a module named 'capped_executor'
from pdelfin.cappedpool import CappedProcessPoolExecutor

# Define functions at the top level to ensure they are picklable by multiprocessing

def square(x):
    return x * x

def raise_exception():
    raise ValueError("Test exception")

def sleep_and_return(x, sleep_time):
    time.sleep(sleep_time)
    return x

def task(counter, max_counter, counter_lock):
    with counter_lock:
        counter.value += 1
        print(f"Task incrementing counter to {counter.value}")
        if counter.value > max_counter.value:
            max_counter.value = counter.value
    time.sleep(0.5)
    with counter_lock:
        counter.value -= 1
    return True

class TestCappedProcessPoolExecutor(unittest.TestCase):

    def test_basic_functionality(self):
        """Test that tasks are executed and results are correct."""
        with CappedProcessPoolExecutor(max_unprocessed=10, max_workers=4) as executor:
            futures = [executor.submit(square, i) for i in range(10)]
            results = [f.result() for f in futures]
            expected = [i * i for i in range(10)]
            self.assertEqual(results, expected)

    def test_exception_handling(self):
        """Test that exceptions in tasks are properly raised."""
        with CappedProcessPoolExecutor(max_unprocessed=10, max_workers=4) as executor:
            future = executor.submit(raise_exception)
            with self.assertRaises(ValueError):
                future.result()

    def test_cancellation(self):
        """Test that tasks can be cancelled before execution."""
        with CappedProcessPoolExecutor(max_unprocessed=10, max_workers=4) as executor:
            future = executor.submit(time.sleep, 5)
            # Try to cancel immediately
            cancelled = future.cancel()
            self.assertTrue(cancelled)
            self.assertTrue(future.cancelled())
            # Attempt to get result; should raise CancelledError
            with self.assertRaises(concurrent.futures.CancelledError):
                future.result()

    def test_shutdown(self):
        """Test that the executor shuts down properly and does not accept new tasks."""
        executor = CappedProcessPoolExecutor(max_unprocessed=10, max_workers=4)
        future = executor.submit(time.sleep, 1)
        executor.shutdown(wait=True)
        with self.assertRaises(RuntimeError):
            executor.submit(time.sleep, 1)

    def test_capping_behavior(self):
        """Test that the number of concurrent tasks does not exceed max_unprocessed."""
        max_unprocessed = 3
        with CappedProcessPoolExecutor(max_unprocessed=max_unprocessed, max_workers=10) as executor:
            from multiprocessing import Manager

            manager = Manager()
            counter = manager.Value('i', 0)
            max_counter = manager.Value('i', 0)
            counter_lock = manager.Lock()

            futures = [executor.submit(task, counter, max_counter, counter_lock) for _ in range(10)]

            for index, f in enumerate(futures):
                print(f"Future {index} returned {f.result()}")

                time.sleep(1)

            print(max_counter.value)
            self.assertLessEqual(max_counter.value, max_unprocessed)

    def test_submit_after_shutdown(self):
        """Test that submitting tasks after shutdown raises an error."""
        executor = CappedProcessPoolExecutor(max_unprocessed=10, max_workers=4)
        executor.shutdown(wait=True)
        with self.assertRaises(RuntimeError):
            executor.submit(square, 2)


if __name__ == '__main__':
    unittest.main()
