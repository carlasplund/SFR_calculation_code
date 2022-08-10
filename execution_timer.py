import time


def execution_timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        val = func(*args, **kwargs)
        print(f"Function '{func.__name__:s}' took {time.time() - t0:.3f} s to execute.")
        return val

    return wrapper


if __name__ == '__main__':

    # Example of use. Put the decorator in front of the function
    # to be timed, like this:

    @execution_timer
    def run(t):
        time.sleep(t)


    run(3)
