'''
A bunch of random utility functions
'''

from counter import Counter
from pprint import pformat

try:
	from itertools import permutations
except ImportError:
	# Backported from python 2.6
	def permutations(iterable, r=None):
		# permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
		# permutations(range(3)) --> 012 021 102 120 201 210
		pool = tuple(iterable)
		n = len(pool)
		r = n if r is None else r
		indices = range(n)
		cycles = range(n, n-r, -1)
		yield tuple(pool[i] for i in indices[:r])
		while n:
			for i in reversed(range(r)):
				cycles[i] -= 1
				if cycles[i] == 0:
					indices[i:] = indices[i+1:] + indices[i:i+1]
					cycles[i] = n - i
				else:
					j = cycles[i]
					indices[i], indices[-j] = indices[-j], indices[i]
					yield tuple(pool[i] for i in indices[:r])
					break
			else:
				return
def getattr_(obj, name, default_thunk):
    "Similar to .setdefault in dictionaries."
    try:
        return getattr(obj, name)
    except AttributeError:
        default = default_thunk()
        setattr(obj, name, default)
        return default


def counted(func, *args):
	def wrapper(*args, **kwargs):
		dic = getattr_(func, "counting_dic", Counter)

		if 'print_counts' in args:
			return pformat(dic)

		# counting_dic is created at the first call
		dic[args[1:]] += 1
		return func(*args)

	return wrapper

def memoized(func, *args):
	def wrapper(*args, **kwargs):
		dic = getattr_(func, "memoize_dic", dict)
		# memoize_dic is created at the first call
		if args in dic:
			return dic[args]
		else:
			result = func(*args)
			dic[args] = result
			return result

	return wrapper
