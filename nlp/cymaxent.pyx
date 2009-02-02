cdef extern from "math.h":
	double exp(double num)

from counter import Counter
from countermap import CounterMap

def log_probs(datum_features, weights, labels):
	log_probs = Counter()

	for label in labels:
		log_probs[label] = label_log_probs(weights[label], datum_features)

	log_probs.log_normalize()

	return log_probs

cdef double label_log_probs(object label_weights, object datum_features):
	cdef double partial_prob = 0.0

	for key in iter(datum_features):
		partial_prob += datum_features[key] * label_weights[key]

	return partial_prob

def expected_counts(labeled_extracted_features, labels, log_probs):
	expected_counts = CounterMap()
	cdef int index = 0
	cdef double cnt = 0.0
	cdef double label_weight = 0.0

	for pair in iter(labeled_extracted_features):
		datum_features = pair[1]

		for label in iter(labels):
			label_weight = exp(log_probs[index][label])
			for feature in iter(datum_features):
				cnt = datum_features[feature]
				expected_counts[label][feature] += label_weight * cnt

		index += 1

	return expected_counts
