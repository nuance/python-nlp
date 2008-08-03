import random
from array import array
from collections import defaultdict
import itertools
import time

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
    
    #### Random access
    print "Random Access"
    rand_access_src = array('i')
    for i in xrange(10000000):
        key = random.choice(key_src)
        rand_access_src.append(key)
    
    last = time.time()
    
    for (container, name) in zip((td, tdd, tl, tda, tla), ("td", "tdd", "tl", "tda", "tla")):
        rand_access(container, rand_access_src)
        print "%s: %f" % (name, time.time()-last)
        last = time.time()
    
    #### Forward Iteration
    print "Iteration Access"
    last = time.time()
    
    for (container, name) in zip((td, tdd, tl, tda, tla), ("td", "tdd", "tl", "tda", "tla")):
        iter_access(container, 1000, 'values' in dir(container))
        print "%s: %f" % (name, time.time()-last)
        last = time.time()
    
    #### Paired iteration
    
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
    test_long_array = array('l', itertools.repeat(0.0, 10000))
    # long_array
    for i in iter_src:
        test_long_array[i] += 1.0
    return test_long_array
        
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