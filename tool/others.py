import random

sub = 0


def name_decorator(func):
    def wrapper(*args, **kwargs):
        global sub
        sub += 1
        print('  ' * sub, '|-', 'In function ', '\"{}{}\"'.format(__file__, func.__name__))
        res = func(*args, **kwargs)
        sub -= 1
        return res

    return wrapper


def print_(*args, **kwargs):
    print('*', '   ' * sub, *args, *kwargs)


def split_list(list_, ratio=0.8):
    list1 = []
    list2 = []
    for e in list_:
        if random.random() < ratio:
            list1.append(e)
        else:
            list2.append(e)
    return list1, list2
