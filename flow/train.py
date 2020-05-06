from flow.data_prepare import get_train_test_collection
from custom import data_prepare, file_filter, multi_prepare_record, multi_generate_record
from tool.others import name_decorator


@name_decorator
def train(args):
    """
    Just train.
    :param args: Just args.
    :return:
    """
    train_collection, test_collection = get_train_test_collection(args.base, data_prepare, file_filter,
                                                                  args.force)

    if args.parallel:
        from flow.data_generator import multiple_prepare, multiple_generator
        train_collection = multiple_prepare(train_collection, multi_prepare_record, args.parallel)
        generator = multiple_generator(train_collection, args.batch, args.parallel, True)
    else:
        from flow.data_generator import single_generator
        generator = single_generator(train_collection, args.batch, multi_prepare_record, multi_generate_record, True)

    import time
    t1 = time.time()
    for _ in range(10):
        res = next(generator)
        print(_)
    print(time.time() - t1)
# collection = data_prepare()
