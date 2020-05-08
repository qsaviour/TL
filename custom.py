import json
import re
import os
from tool.path_parser import path_split, suffix_filter
import cv2
from tool.processor import processor
import numpy as np


def file_filter(data_path):
    return suffix_filter(['.json'])(data_path)


def data_prepare(data_path, collection):
    with open(data_path) as f:
        json_content = json.load(f)
    if json_content.get('shapes') is None or len(json_content.get('shapes')) == 0:
        return
    json_path_base, json_name, prefix, suffix = path_split(data_path)
    image_path_base = re.sub("json[/|\\\]?$", "jpg", json_path_base)
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
        x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
        collection.append(
            {'img_path': image_path, 'annotation': annotation, 'location': [x, y, w, h], 'label': label})
        return collection


def multi_prepare_record(record):
    # record['img'] = cv2.imread(record['img_path'])
    return record


def multi_generate_record(record, augment):
    # img = record['img']
    print(record['img_path'])
    print(record['label'])
    print(record['location'])
    img = cv2.imread(record['img_path'])
    location = record['location']
    x, y, w, h = map(int, location)
    if augment:
        crop = processor.augment(img, location, (128 * 3, 128 * 3), True, 0.05, 0.05)
    else:
        crop = processor.augment(img, location, (128 * 3, 128 * 3), True, 0.05, 0.05, False, False, False, False, False,
                                 False,
                                 False)
    crop = crop.reshape((1,) + crop.shape)

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("ground", img)

    label = record['label']
    label = np.array(label).reshape(1, -1)
    return crop / 255.0, label
