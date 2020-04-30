import autokeras as ak
from flow.data_prepare import pre_prepare, exist_pkl, get_pkl
from custom import data_prepare, file_filter, split_train_test_set, multi_prepare_record, generate_x_y
from tool.path_parser import cvt_abs_path
from tool.others import print_, balance_label

from multiprocessing.dummy import Pool
from multiprocessing import Pool, cpu_count
import numpy as np


def generate_x_y_(obj):
    return generate_x_y(obj, augment=False)


def search_autokeras(args):
    """
    Just train.
    :param args: Just args.
    :return:
    """
    pool = Pool(cpu_count() - 2)
    print(11)
    if exist_pkl(args.base) and not args.force:
        print_("use the existing pkl file")
        train_collection, test_collection = get_pkl(args.base)
    else:
        print_("prepare data and dump into pkl file")
        collection = pre_prepare(cvt_abs_path(args.base), data_prepare, file_filter)
        train_collection, test_collection = split_train_test_set(collection)

    train_collection = pool.map(multi_prepare_record, train_collection)
    test_collection = pool.map(multi_prepare_record, test_collection)

    train_batch = pool.map(generate_x_y_, train_collection)
    test_batch = pool.map(generate_x_y_, test_collection)
    print(99)
    train_batch = np.array(train_batch)
    test_batch = np.array(test_batch)
    clf = ak.ImageClassifier()
