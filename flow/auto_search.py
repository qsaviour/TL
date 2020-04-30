import autokeras as ak
from flow.data_prepare import pre_prepare, exist_pkl, get_pkl
from custom import data_prepare, file_filter, split_train_test_set, multi_prepare_record, generate_x_y
from tool.path_parser import cvt_abs_path, make_dirs
from tool.others import print_, name_decorator,print_

from multiprocessing.dummy import Pool
from multiprocessing import Pool, cpu_count
import numpy as np


def generate_x_y_(obj):
    return generate_x_y(obj, augment=False)


@name_decorator
def search_autokeras(args):
    """
    Just train.
    :param args: Just args.
    :return:
    """
    pool = Pool(cpu_count() - 2)
    if exist_pkl(args.base) and not args.force:
        print_("use the existing pkl file")
        train_collection, test_collection = get_pkl(args.base)
    else:
        print_("prepare data and dump into pkl file")
        collection = pre_prepare(cvt_abs_path(args.base), data_prepare, file_filter)
        train_collection, test_collection = split_train_test_set(collection)

    print('train size:', len(train_collection))
    print('test size:', len(test_collection))

    print_('multi_prepare_record...')
    train_collection = pool.map(multi_prepare_record, train_collection)
    test_collection = pool.map(multi_prepare_record, test_collection)

    print_('generate_x_y_...')
    train_batch = list(zip(pool.map(generate_x_y_, train_collection)))
    test_batch = list(zip(pool.map(generate_x_y_, test_collection)))


    x_train = np.concatenate([e[0][0] for e in train_batch])
    y_train = np.concatenate([e[0][1] for e in train_batch])

    x_test = np.concatenate([e[0][0] for e in test_batch])
    y_test = np.concatenate([e[0][1] for e in test_batch])

    clf = ak.ImageClassifier(max_trials=10)
    clf.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)
    model = clf.export_model()
    target_folder = '/'.join([args.base, 'stock'])
    make_dirs(target_folder)
    target_path = '/'.join([target_folder, 'best.h5'])
    model.save(target_path)
