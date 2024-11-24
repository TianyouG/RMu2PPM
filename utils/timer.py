import time
import logging


def timing_decorator(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"{func.__name__} execution time: {elapsed_time:.6f} seconds")
        logging.info(f"{func.__name__} execution time: {elapsed_time:.6f} seconds")
        return result, elapsed_time

    return wrapper
