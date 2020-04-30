from flow.data_prepare import pre_prepare, exist_pkl, get_pkl, save_pkl
from custom import data_prepare, file_filter, split_train_test_set
from tool.path_parser import cvt_abs_path
from tool.others import name_decorator, print_


@name_decorator
def train(args):
    """
    Just train.
    :param args: Just args.
    :return:
    """
    if exist_pkl(args.base) and not args.force:
        print_("use the existing pkl file")
        train_collection, test_collection = get_pkl(args.base)
    else:
        print_("prepare data and dump into pkl file")
        collection = pre_prepare(cvt_abs_path(args.base), data_prepare, file_filter)
        train_collection, test_collection = split_train_test_set(collection)
        save_pkl(args.base, train_collection, test_collection)

    if args.single:
        from flow.data_generator import single_generator
        generator = single_generator(train_collection, args.batch, args.orderly_sample)
    else:
        from flow.data_generator import multiple_generator
        generator = multiple_generator(train_collection, args.batch, args.orderly_sample)

    import time
    t1=time.time()
    for _ in range(100):
        res = next(generator)
        print(_)
    print(time.time()-t1)
# collection = data_prepare()
