from flow.data_prepare import get_train_test_collection
from custom import data_prepare, file_filter, split_train_test_set, multi_prepare_record, generate_x_y
from tool.path_parser import make_dirs
from tool.others import name_decorator, print_

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
import autokeras as ak


def generate_x_y_(obj):
    return generate_x_y(obj, augment=False)


@name_decorator
def search_autokeras(args):
    """
    Just train.
    :param args: Just args.
    :return:
    """
    train_collection, test_collection = get_train_test_collection(args.base, data_prepare, file_filter,
                                                                  split_train_test_set, args.force)

    if args.parallel:
        from multiprocessing.dummy import Pool
        from multiprocessing import Pool, cpu_count
        pool = Pool(cpu_count() - 2)

        train_collection = pool.map(multi_prepare_record, train_collection)
        train_collection = list(zip(pool.map(generate_x_y_, train_collection)))

        test_collection = pool.map(multi_prepare_record, test_collection)
        test_collection = list(zip(pool.map(generate_x_y_, test_collection)))

        x_train = np.concatenate([e[0][0] for e in train_collection])
        y_train = np.concatenate([e[0][1] for e in train_collection])

        x_test = np.concatenate([e[0][0] for e in test_collection])
        y_test = np.concatenate([e[0][1] for e in test_collection])
    else:

        train_collection = [multi_prepare_record(e) for e in train_collection]
        train_collection = [generate_x_y_(e) for e in train_collection]

        test_collection = [multi_prepare_record(e) for e in test_collection]
        test_collection = [generate_x_y_(e) for e in test_collection]

        x_train = np.concatenate([e[0] for e in train_collection])
        y_train = np.concatenate([e[1] for e in train_collection])

        x_test = np.concatenate([e[0] for e in test_collection])
        y_test = np.concatenate([e[1] for e in test_collection])

    clf = ak.ImageClassifier(max_trials=10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    clf.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)
    model = clf.export_model()
    target_folder = '/'.join([args.base, 'stock'])
    make_dirs(target_folder)
    target_path = '/'.join([target_folder, 'best.h5'])
    model.save(target_path)
