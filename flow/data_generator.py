from custom import generate
from tool.others import name_decorator, print_
import random


@name_decorator
def single_generator(collection, batch_size, gen_type):
    pass


@name_decorator
def multiple_generator(collection, batch_size, gen_type, feed_type='1in1', q_limit=100):
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
        collection_batch = random.choices(collection, k=batch_size)
        queue.put(1)
        print("queue size", queue.qsize())

    def feed_queue3():
        pass

    from multiprocessing.dummy import Pool, JoinableQueue
    from multiprocessing import cpu_count, Process

    assert feed_type in ('1in1', '3in1')

    if feed_type == '1in1':
        queue = JoinableQueue()
        pool = Pool(cpu_count() - 2)

