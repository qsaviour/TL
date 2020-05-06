import random

sub = 0


def name_decorator(func):
    def wrapper(*args, **kwargs):
        global sub
        sub += 1
        print('  ' * sub, '|-', 'In function ',
              '\"{}.{}\"'.format(func.__module__, func.__name__))
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


@name_decorator
def balance_label(collection_, key):
    label_set = dict()
    for e in collection_:
        label_set[e[key]] = label_set.get(e[key], 0) + 1
    print_("data distribute :", label_set)
    collection_filters = []
    for label in label_set:
        collection_filters.append(list(filter(lambda z: z[key] == label, collection_)))
    max_num = max([len(list(e)) for e in collection_filters])
    collection_res = []
    for i, collection_filter in enumerate(collection_filters):
        n = len(list(collection_filter))
        quotient = max_num // n
        remain = max_num % n
        collection_tmp = list(collection_filter)
        collection_tmp = collection_tmp * quotient + collection_tmp[:remain]

        collection_res += collection_tmp

    return collection_res
