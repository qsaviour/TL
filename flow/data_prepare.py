import os
from tool.path_parser import dfs, cvt_abs_path, make_dirs
import pickle
from tool.others import name_decorator, print_, split_train_test_collection
import random


def get_train_test_collection(base, data_prepare, file_filter, force=True):
    if exist_pkl(base) and not force:
        print_("use the existing pkl file")
        train_collection, test_collection = get_pkl(base)
    else:
        print_("prepare data and dump into pkl file")
        data_path = cvt_abs_path(os.path.join(cvt_abs_path(base), 'data'))
        collection = []
        dfs(data_path, file_filter, data_prepare, collection)
        train_collection, test_collection = split_train_test_collection(collection)
        save_pkl(base, train_collection, test_collection)

    random.shuffle(train_collection)
    random.shuffle(test_collection)
    print_('train size:', len(train_collection))
    print_('test size:', len(test_collection))

    return train_collection, test_collection


@name_decorator
def exist_pkl(base_path):
    stock_path = cvt_abs_path(os.path.join(base_path, 'stock'))
    train_pkl_path = os.path.join(stock_path, 'train_collection.pkl')
    test_pkl_path = os.path.join(stock_path, 'test_collection.pkl')
    if not os.path.exists(train_pkl_path) or not os.path.exists(test_pkl_path):
        return False
    else:
        return True


@name_decorator
def get_pkl(base_path):
    stock_path = cvt_abs_path(os.path.join(base_path, 'stock'))
    train_pkl_path = os.path.join(stock_path, 'train_collection.pkl')
    test_pkl_path = os.path.join(stock_path, 'test_collection.pkl')

    with open(train_pkl_path, 'rb') as f:
        train_collection = pickle.load(f)
    with open(test_pkl_path, 'rb') as f:
        test_collection = pickle.load(f)

    return train_collection, test_collection


@name_decorator
def save_pkl(base_path, train_collection, test_collection):
    # make stock dir
    stock_path = cvt_abs_path(os.path.join(base_path, 'stock'))
    make_dirs(stock_path)

    # train/test pkl path
    train_pkl_path = os.path.join(stock_path, 'train_collection.pkl')
    test_pkl_path = os.path.join(stock_path, 'test_collection.pkl')

    print("write paths to pkl:", train_pkl_path, test_pkl_path)

    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_collection, f)
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test_collection, f)
