from functools import partial
import multiprocessing


def f(a, b, c, d):
    print("{} {} {}".format(a, b, c, d))


def main():
    iterable = range(100000)  # [1, 2, 3, 4, 5]
    pool = multiprocessing.Pool()
    a = "hi"
    b = "there"
    d = list(range(100000))
    func = partial(f, a, b, d)
    pool.map(func, iterable)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
