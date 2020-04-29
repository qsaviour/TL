from custom import generate
from tool.others import name_decorator, print_
import random
import cv2


@name_decorator
def single_generator(collection, batch_size, gen_type):
    pass


@name_decorator
def multiple_generator(collection, batch_size, gen_type, feed_type='1in1', q_limit=10):
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
                for e in collection_batch:
                    queue.put([generate(e) for e in collection_batch])


    def feed_queue3():
        pass

    from multiprocessing.dummy import Pool, JoinableQueue, Process
    from multiprocessing import cpu_count

    assert feed_type in ('1in1', '3in1')

    if feed_type == '1in1':
        queue = JoinableQueue()
        p_num = max(cpu_count() - 2, 1)
        pool = Pool(p_num)
        pool.apply_async(feed_queue1)
        pool.apply_async(feed_queue1)
        pool.apply_async(feed_queue1)
        # process = Process(target=feed_queue1)
        # process.run()

        # while True:
        #     yield queue.get()
