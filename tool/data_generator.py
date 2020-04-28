import os
import re
from tool.path_parser import dfs, suffix_filter, path_split
from multiprocessing.dummy import Pool, Process
from multiprocessing import cpu_count
from processor import processor
import cv2
import json
import time

pool = Pool(cpu_count() * 2 // 3)

def prepare_data(json_path_, collection_):
    with open(json_path_) as f:
        json_content = json.load(f)
    if json_content.get('shapes') is None or len(json_content.get('shapes')) == 0:
        return
    json_path_base, json_name, prefix, suffix = path_split(json_path_)
    image_path_base = re.sub("json[/|\\\]?$", "jpg", json_path_base)
    image_path = '/'.join([image_path_base, json_content["imagePath"]])
    if not os.path.exists(image_path):
        print(image_path, "not exists")
    label = json_content['shapes'][0]['label']
    location = json_content['shapes'][0]['points']
    (x, y), (w, h) = location
    collection_.append([image_path, label, {'location': [x, y, w, h]}])


def read_image_from_collection(collection, limit_size=None):

    def read_a(a_record):
        global q
        img = cv2.imread(a_record[0])
        q.put(img)
        # return [img, ] + a_record[1:]

    n = len(collection)
    if type(limit_size) is int:
        n = min(limit_size, n)
    t1 = time.time()
    collection = pool.map(read_a, collection[:n])  # [image_path,label]
    print(time.time() - t1)

    return collection


def data_generator(data_path, size, batch_size):
    collection = []
    dfs(data_path, suffix_filter(['.json']), prepare_data, collection)
    collection = read_image_from_collection(collection, 64)
    return collection


def read_(a_record):
    img = cv2.imread(a_record[0])
    return a_record + [processor.augment(img), ]


if __name__ == '__main__':

    c = data_generator('../../data', 0, 0)
