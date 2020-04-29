from tool.path_parser import dfs, suffix_filter
import json
import os


def do_function(path, collection):
    with open(path) as f:
        json_content = json.load(f)
    if json_content['imageData']:
        del json_content['imageData']
    with open(path, 'w') as f:
        json.dump(json_content, f)
    print(path)

dfs('../data', suffix_filter(['.json']), do_function, None)
