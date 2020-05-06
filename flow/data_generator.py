from custom import multi_prepare_record, multi_generate_record
from tool.others import name_decorator
import random
import numpy as np


@name_decorator
def single_generator(collection, batch_size, aug):
    while True:
        collection_batch = random.choices(collection, k=batch_size)
        x = []
        y = []
        for e in collection_batch:
            e = multi_prepare_record(e)
            x_, y_ = multi_generate_record(e, aug)
            x.append(x_)
            y.append(y_)
        yield x, y


def multiple_prepare(collection, function, worker_num):
    from multiprocessing.dummy import Pool
    from multiprocessing import cpu_count
    p_num = max(min(worker_num, cpu_count() - 1), 1)
    pool = Pool(p_num)
    collection = pool.map(function, collection)
    pool.close()
    del pool
    return collection


@name_decorator
def multiple_generator(collection, batch_size, worker_num, aug, q_limit=10):
    """
    :param collection: collection
    :param batch_size:
    :param gen_type:
    :param feed_type: 1in1 - one thread feeds data to one batch.
                      3in1 - multiple threads feed data to one batch (may be need used in generate one big batch).
    :param q_limit:
    :return:
    """

    @name_decorator
    def feed_queue1():
        while True:
            if queue.qsize() < q_limit:
                collection_batch = random.choices(collection, k=batch_size)
                x = []
                y = []
                for e in collection_batch:
                    x_, y_ = multi_generate_record(e, aug)
                    x.append(x_)
                    y.append(y_)
                print(1)
                x = np.concatenate(x, axis=0)
                print(2)
                y = np.concatenate(y, axis=0)
                print(3)
                queue.put((x, y))

    def feed_queue3():
        pass

    from multiprocessing.dummy import Pool, JoinableQueue
    from multiprocessing import cpu_count

    queue = JoinableQueue()
    p_num = max(min(worker_num, cpu_count() - 1), 1)
    pool = Pool(p_num)
    for _ in range(p_num):
        pool.apply_async(feed_queue1)

    while True:
        out = queue.get()
        yield out
