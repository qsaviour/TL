import json
import re
import os
from tool.path_parser import path_split, suffix_filter
from tool.others import split_list, name_decorator, balance_label
import cv2
from tool.processor import processor


def file_filter(data_path):
    """
    Whether  The data_path_ will be passed into "data_prepare" function.
    :param data_path: the right one.
    :return: True/False
    """
    return suffix_filter(['.json'])(data_path)


def data_prepare(data_path, collection):
    """
    Be done in dfs. Be done on single process.
    :param data_path: [data_folder]
    :param collection: [ a_record]. a_record - store the information
    :return: prepared collection. The collection will be passed into "split_train_test_set" function
    """
    with open(data_path) as f:
        json_content = json.load(f)
    if json_content.get('shapes') is None or len(json_content.get('shapes')) == 0:
        return
    json_path_base, json_name, prefix, suffix = path_split(data_path)
    image_path_base = re.sub("json[/|\\\]?$", "jpg", json_path_base)
    # image_path = '/'.join([image_path_base, os.path.split(json_content["imagePath"])[-1]])
    image_path = '/'.join([image_path_base, prefix + '.jpg'])
    if not os.path.exists(image_path):
        print(image_path, "not exists")
        return
    for a_data in json_content['shapes']:
        annotation = a_data['label']
        if annotation == 'tesla':
            label = [1, 0]
        else:
            annotation = 'others'
            label = [0, 1]
        location = a_data['points']
        (x1, y1), (x2, y2) = location
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        collection.append(
            {'img_path': image_path, 'annotation': annotation, 'location': [x, y, w, h], 'label': label})
        return collection


@name_decorator
def split_train_test_set(collection):
    train_collection, test_collection = split_list(collection, 0.8)

    train_collection = balance_label(train_collection, 'annotation')
    test_collection = balance_label(test_collection, 'annotation')

    return train_collection, test_collection


def multi_prepare_record(record):
    record['img'] = cv2.imread(record['img_path'])
    return record


def generate_x_y(record, augment):
    img = record['img']
    # img = cv2.imread(record['img_path'])
    location = record['location']
    if augment:
        crop = processor.augment(img, location, (256, 256), True, 0.05, 0.05, False, False, False, False, False, False,
                                 False)
    else:
        crop = processor.augment(img, location, (256, 256), True, 0.05, 0.05, False, False, False, False, False, False,
                                 False)
    crop = crop.reshape((1,) + crop.shape)
    label = record['label']
    return crop / 255.0, label
