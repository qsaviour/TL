from flow.data_prepare import get_train_test_collection
from custom import data_prepare, file_filter

train_collection, test_collection = get_train_test_collection('../', data_prepare, file_filter,
                                                              True)
from flow.data_generator import multiple_prepare, multiple_generator

test_collection = multiple_prepare(test_collection, 3)
g_validate = multiple_generator(test_collection, 32, 3, False, randomly=False)
x, y = next(g_validate)
