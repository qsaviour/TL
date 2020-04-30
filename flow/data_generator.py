from custom import generate_x_y
from tool.others import name_decorator, print_
import random
import cv2


@name_decorator
def single_generator(collection, batch_size, gen_type):
    pass


@name_decorator
def multiple_generator(collection, batch_size, worker_num, gen_type, feed_type='1in1', q_limit=10):
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
                    x_, y_ = generate_x_y(e)
                    x.append(x_)
                    y.append(y_)
                queue.put((x, y))

    def feed_queue3():
        pass

    from multiprocessing.dummy import Pool, JoinableQueue
    from multiprocessing import cpu_count

    assert feed_type in ('1in1', '3in1')

    if feed_type == '1in1':
        queue = JoinableQueue()
        p_num = max(min(worker_num, cpu_count() - 1), 1)
        pool = Pool(p_num)
        for _ in range(p_num):
            pool.apply_async(feed_queue1)

        while True:
            yield queue.get()
