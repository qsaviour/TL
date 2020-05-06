from multiprocessing import Pool, Process
import time


def k():
    print(9)


if __name__ == '__main__':
    pool = Pool(3)
    for _ in range(100):
        pool.apply_async(k)
    time.sleep(2)
    print(4)
