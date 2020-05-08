from custom import multi_prepare_record, multi_generate_record
from tool.others import name_decorator
import numpy as np


@name_decorator
def single_generator(collection, batch_size, aug):
    while True:
        collection_batch = np.random.choice(collection, batch_size)
        x = []
        y = []
        for e in collection_batch:
            e = multi_prepare_record(e)
            x_, y_ = multi_generate_record(e, aug)
            x.append(x_)
            y.append(y_)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        yield x, y
    # return np.random.random((32, 256, 256, 3)), np.random.randint(0, 2, (32, 2))


def multiple_prepare(collection, worker_num):
    from multiprocessing.dummy import Pool
    from multiprocessing import cpu_count
    p_num = max(min(worker_num, cpu_count() - 1), 1)
    pool = Pool(p_num)
    collection = pool.map(multi_prepare_record, collection)
    pool.close()
    del pool
    return collection


@name_decorator
def multiple_generator(collection, batch_size, worker_num, aug, q_limit=10, randomly=True):
    @name_decorator
    def feed_queue1():
        while True:
            if queue.qsize() < q_limit:
                collection_batch = np.random.choice(collection, batch_size)
                x = []
                y = []
                for e in collection_batch:
                    x_, y_ = multi_generate_record(e, aug)
                    x.append(x_)
                    y.append(y_)
                x = np.concatenate(x, axis=0)
                y = np.concatenate(y, axis=0)
                queue.put((x, y))

    def feed_queue3():
        pass

    from multiprocessing.dummy import Pool, JoinableQueue
    from multiprocessing import cpu_count
    p_num = max(min(worker_num, cpu_count() - 1), 1)
    pool = Pool(p_num)
    if randomly:
        queue = JoinableQueue()
        for _ in range(p_num):
            pool.apply_async(feed_queue1)

        while True:
            out = queue.get()
            yield out
    else:
        f = lambda e: multi_generate_record(e, aug)
        out = pool.map(f, collection)
        x = np.concatenate([e[0] for e in out], axis=0)
        y = np.concatenate([e[1] for e in out], axis=0)
        while True:
            yield x, y
