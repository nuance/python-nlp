from itertools import izip, repeat, chain

from maxent import get_log_probabilities, get_expected_counts
from countermap import CounterMap
from counter import Counter

def cnter(l):
	return Counter(izip(l, repeat(1.0, len(l))))

training_data = (('cat', cnter(('fuzzy', 'claws', 'small'))),
				 ('bear', cnter(('fuzzy', 'claws', 'big'))),
				 ('cat', cnter(('claws', 'medium'))))

labels = set([label for label, _ in training_data])
features = set()
for _, counter in training_data:
	features.update(set(counter.keys()))

weights = CounterMap()

log_probs = list()
for pos, (label, features) in enumerate(training_data):
	log_probs.append(get_log_probabilities(features, weights, labels))

test = get_expected_counts(training_data, labels, log_probs, CounterMap())

print test
