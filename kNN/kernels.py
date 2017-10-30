
import math


# https://en.wikipedia.org/wiki/Kernel_(statistics)
kernels = {
    'epanechnikov': lambda val: 0.75 * (1 - val * val),
    'gaussian': lambda val: 1 / math.sqrt(2 * math.pi) * math.exp((-1 / 2) * val * val),
    # 'uniform': lambda val: 1 / 2,
    # 'triangular': lambda val: 1 - abs(val),
    'quartic': lambda val: 15 / 16 * (1 - val ** 2) ** 2
}