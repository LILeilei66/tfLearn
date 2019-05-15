import itertools

# todo

DISTANCE_WEIGHT_NAME = 'distance'

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

if __name__ == '__main__':
    result = pairwise([1,5,2,6,3])
    for value in result:
        print(value)
