from math import exp
import time

from nlp import counter
from maxent import get_log_probabilities as log_probs
from countermap import CounterMap

def slow_log_probs(datum_features, weights, labels):
	log_probs = counter((label, sum((weights[label] * datum_features).itervalues())) for label in labels)
	log_probs.log_normalize()
	log_probs.default = float("-inf")
	return log_probs

features = counter((key, 1.0) for key in ['warm', 'fuzzy'])

weights = CounterMap()
weights['dog'] = counter({'warm' : 2.0, 'fuzzy' : 0.5})
weights['cat'] = counter({'warm' : 0.5, 'fuzzy' : 2.0})

labels = set(weights.iterkeys())

logp = log_probs(features, weights, labels)

print "** Log probs **"
print logp
print slow_log_probs(features, weights, labels)
print "** Regular probs **"
print counter((key, exp(val)) for key, val in logp.iteritems())

assert(abs(sum(exp(val) for val in logp.itervalues()) - 1.0) < 0.00001)

start = time.time()
for i in xrange(100000):
	test = slow_log_probs(features, weights, labels)

print "slow time: %f" % (time.time() - start)

start = time.time()
for i in xrange(100000):
	test = log_probs(features, weights, labels)

print "fast time: %f" % (time.time() - start)

