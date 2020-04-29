import os
import shutil


def make_dirs(path, force=False):
    if force:
        if os.path.exists(path):
            shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def cvt_abs_path(rpath):
    return os.path.normpath(os.path.join(os.getcwd(), rpath))


def dfs(path, filter_function, do_function, collection, collection_limit=None):
    print('* In folder', path)
    for obj in os.listdir(path):
        if type(collection_limit) is int:
            if len(collection) > collection_limit:
                return
        cur_path = '/'.join([path, obj])
        if os.path.isdir(cur_path) and 'invalid' not in cur_path.lower():
            dfs(cur_path, filter_function, do_function, collection, collection_limit=collection_limit)
        elif filter_function(cur_path):
            do_function(cur_path, collection)


def suffix_filter(suffix_l):
    def t_suffix(path):
        if os.path.splitext(path)[1] in suffix_l:
            return True
        else:
            return False

    return t_suffix


image_filter = suffix_filter(['.jpg', '.png', '.jpeg'])


def path_split(path):
    base_path, file_name = os.path.split(path)
    prefix, suffix = os.path.splitext(file_name)
    return base_path, file_name, prefix, suffix


if __name__ == '__main__':
    test_path = 'tool/path_parser.py'
    print(image_filter(test_path))
    print(path_split(test_path))
    dfs('..', suffix_filter(['.py']), print, '')
