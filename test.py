from multiprocessing.dummy import Pool


def k(i):
    for _ in range(9):
        print(8)
    return 2


def A():
    print('A')
    pool = Pool(2)
    pool.map(k,range(2))



def B():
    print('B')
    A()


if __name__ == "__main__":
    B()
