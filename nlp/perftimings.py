import random
from array import array
from collections import defaultdict
import itertools
import time

from counter import Counter
from countermap import CounterMap
from nlp import counter

from numpy.core import array as np_array
from numpy.core import divide

test_iteration = False

def main():
	key_src = range(10000)
	iter_src = array('i')

	# initialize random value/key initialization ordering
	for i in xrange(100000):
		key = random.choice(key_src)
		iter_src.append(key)

	#### Initialization
	print "Timing random initialization"

	last = time.time()
	td = dict_init(iter_src)
	print "%s: %f" % ("td", time.time()-last)
	last = time.time()
	tdd = defaultdict_init(iter_src)
	print "%s: %f" % ("tdd", time.time()-last)
	last = time.time()
	tl = list_init(iter_src)
	print "%s: %f" % ("tl", time.time()-last)
	last = time.time()
	tda = double_array_init(iter_src)
	print "%s: %f" % ("tda", time.time()-last)
	last = time.time()
	tla = long_array_init(iter_src)
	print "%s: %f" % ("tla", time.time()-last)
	last = time.time()
	cnt = counter_init(iter_src)
	print "%s: %f" % ("counter", time.time()-last)
	last = time.time()
	cCnt = cCounter_init(iter_src)
	print "%s: %f" % ("cCounter", time.time()-last)
	last = time.time()
	npa = npa_init(iter_src)
	print "%s: %f" % ("npa", time.time()-last)

	if test_iteration:

#### Random access
			print "Random Access"
			rand_access_src = array('i')
			for i in xrange(10000000):
				key = random.choice(key_src)
				rand_access_src.append(key)

				last = time.time()

			for (container, name) in zip((td, tdd, tl, tda, tla, cnt, cCnt, npa), ("td", "tdd", "tl", "tda", "tla", "cnt", "cCnt", "npa")):
				rand_access(container, rand_access_src)
				print "%s: %f" % (name, time.time()-last)
				last = time.time()

			#### Forward Iteration
			print "Iteration Access"
			last = time.time()

			for (container, name) in zip((td, tdd, tl, tda, tla, cnt, cCnt, npa), ("td", "tdd", "tl", "tda", "tla", "cnt", "cCnt", "npa")):
				iter_access(container, 1000, 'values' in dir(container))
				print "%s: %f" % (name, time.time()-last)
				last = time.time()

			#### Random-stride forward iteration (simulates sorted sparse index)
			print "Sparse Iteration Access"
			stride_src = sorted(rand_access_src)
			last = time.time()

			for (container, name) in zip((td, tdd, tl, tda, tla, cnt, cCnt, npa), ("td", "tdd", "tl", "tda", "tla", "cnt", "cCnt", "npa")):
				for pos in stride_src:
					temp = container[pos]
				print "%s: %f" % (name, time.time()-last)
				last = time.time()

			#### Random-stride backward-forward iteration (simulates sorted sparse index)
			print "Sparse Forward/Backward Iteration Access"
			back = list(xrange(len(stride_src)))
			for i in xrange(len(stride_src)):
				back[i] = random.choice([True,False])

			access_order = list()
			current_pos = 0
			for (pos, stride) in enumerate(stride_src):
				if back[pos]: current_pos = (current_pos - stride) % len(container)
				else: current_pos = (current_pos + stride) % len(container)
				access_order.append(current_pos)

			last = time.time()

			for (container, name) in zip((td, tdd, tl, tda, tla, cnt, cCnt, npa), ("td", "tdd", "tl", "tda", "tla", "cnt", "cCnt", "npa")):
				for pos in access_order:
					temp = container[pos]
				print "%s: %f" % (name, time.time()-last)
				last = time.time()
		
	#### Test counter-specifics
	print "arg_max"
	last = time.time()
	
	for (container, name) in zip((cnt, cCnt, npa), ("cnt", "cCnt")):
		temp = container.arg_max()
		print "%s: %f (%s)" % (name, time.time()-last, temp)
		last = time.time()

	last = time.time()
	temp = npa.argmax()
	print "%s: %f (%s)" % ("npa", time.time()-last, temp)
	last = time.time()
	
	print "total_count"
	last = time.time()

	for (container, name) in zip((cnt, cCnt), ("cnt", "cCnt")):
		temp = container.total_count()
		print "%s: %f (%f)" % (name, time.time()-last, temp)
		last = time.time()

	last = time.time()
	temp = npa.sum()
	print "%s: %f (%f)" % ("npa", time.time()-last, temp)
	last = time.time()
		
	print "normalize"
	last = time.time()

	for (container, name) in zip((cnt, cCnt), ("cnt", "cCnt")):
		container.normalize()
		print "%s: %f" % (name, time.time()-last)
		last = time.time()

	last = time.time()
	npa = divide(npa, npa.sum())
	print "%s: %f" % ("npa", time.time()-last)
	last = time.time()

	print "arg_max"
	last = time.time()

	for (container, name) in zip((cnt, cCnt), ("cnt", "cCnt")):
		temp = container.arg_max()
		print "%s: %f (%s)" % (name, time.time()-last, temp)
		last = time.time()

	last = time.time()
	temp = npa.argmax()
	print "%s: %f (%s)" % ("npa", time.time()-last, temp)
	last = time.time()
		
	print "total_count"
	last = time.time()

	for (container, name) in zip((cnt, cCnt), ("cnt", "cCnt")):
		temp = container.total_count()
		print "%s: %f (%f)" % (name, time.time()-last, temp)
		last = time.time()

	last = time.time()
	temp = npa.sum()
	print "%s: %f (%f)" % ("npa", time.time()-last, temp)
	last = time.time()

def dict_init(iter_src):
	print "dict"
	test_dict = dict(itertools.izip(xrange(10000), itertools.repeat(0.0, 10000)))

	# dict
	for i in iter_src:
		test_dict[i] += 1.0
	return test_dict

def defaultdict_init(iter_src):
	test_defaultdict = defaultdict(lambda:0.0)
	# defaultdict
	for i in iter_src:
		test_defaultdict[i] += 1.0
	return test_defaultdict

def list_init(iter_src):
	test_list = list(itertools.repeat(0.0, 10000))
	# list
	for i in iter_src:
		test_list[i] += 1.0
	return test_list

def double_array_init(iter_src):
	test_double_array = array('d', itertools.repeat(0.0, 10000))
	# double_array
	for i in iter_src:
		test_double_array[i] += 1.0
	return test_double_array

def long_array_init(iter_src):
	test_long_array = array('l', itertools.repeat(0, 10000))
	# long_array
	for i in iter_src:
		test_long_array[i] += 1
	return test_long_array

def counter_init(iter_src):
	test_counter = Counter()
	for i in iter_src:
		test_counter[i] += 1
	return test_counter

def cCounter_init(iter_src):
	test_counter = counter()
	for i in iter_src:
		test_counter[i] += 1
	return test_counter

def countermap_init(iter_src):
	test_countermap = CounterMap()
	for i in iter_src:
		test_countermap[i] += 1
	return test_countermap

def npa_init(iter_src):
	test_numpy_array = np_array(list(itertools.repeat(0.0, 10000)))
	# numpy_array
	for i in iter_src:
		test_numpy_array[i] += 1.0
	return test_numpy_array

def rand_access(container, iteration):
	temp_value = 0.0
	for key in iteration:
		temp_value = container[key]

def iter_access(container, count, values=False):
	temp_value = 0.0
	if values:
		for i in xrange(count):
			for i in container:
				temp_value = container[i]
		return
	for i in xrange(count):
		for i in container:
			temp_value = i

if __name__ == "__main__":
	main()
