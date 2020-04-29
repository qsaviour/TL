import json
import re
import os
from tool.path_parser import path_split, suffix_filter
from tool.others import split_list
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
    image_path = '/'.join([image_path_base, os.path.split(json_content["imagePath"])[-1]])
    if not os.path.exists(image_path):
        print(image_path, "not exists")
        return
    for a_data in json_content['shapes']:
        label = a_data['label']
        location = a_data['points']
        (x, y), (w, h) = location
        collection.append({'img_path': image_path, 'label': label, 'location': [x, y, w, h]})
    return collection


def split_train_test_set(collection):
    train_collection, test_collection = split_list(collection, 0.8)
    return train_collection, test_collection


def generate_x_y(record):
    img = cv2.imread(record['img_path'])
    location = record['location']
    crop = processor.augment(img, location, target_shape=(200, 200))
    print(crop.shape)
    if record['label'] == 'tesla':
        label = [1, 0]
    else:
        label = [0, 1]
    return crop, label
