from __future__ import with_statement

__maxent_functions__ = "C"

from math import exp

# python modules
from countermap import CounterMap
from counter import Counter
from features import ngrams
from function import Function
from minimizer import Minimizer
from itertools import izip, repeat

def slow_log_probs(datum_features, weights, labels):
	log_probs = Counter((label, sum((weights[label] * datum_features).itervalues())) for label in labels)
	log_probs.log_normalize()
	log_probs.default = float("-inf")

	return log_probs

def slow_expected_counts(labeled_extracted_features, labels, log_probs):
	expected_counts = CounterMap()

	for (index, (_, datum_features)) in enumerate(labeled_extracted_features):
		for (feature, cnt) in datum_features.iteritems():
			for label in labels:
				expected_counts[label][feature] += exp(log_probs[index][label]) * cnt

	return expected_counts

if __maxent_functions__ == "C":
	# c modules
	from maxent import get_log_probabilities as get_log_probs
	from maxent import get_expected_counts
elif __maxent_functions__ == "cython":
	import cymaxent
	get_log_probs = cymaxent.log_probs
	get_expected_counts = lambda a, b, c, d: cymaxent.expected_counts(a, b, c)
else:
	get_log_probs = slow_log_probs
	get_expected_counts = lambda a, b, c, d: slow_expected_counts(a, b, c)

class MaxEntWeightFunction(Function):
	sigma = 1.0
	labels = None
	features = None
	empirical_counts = None

	def __init__(self, labeled_extracted_features, labels, features):
		self.labeled_extracted_features = labeled_extracted_features
		self.labels = labels
		self.features = features
		self.empirical_counts = CounterMap()

		print "Calculating empirical counts..."
		
		for (index, (datum_label, datum_features)) in enumerate(self.labeled_extracted_features):
			for (feature, cnt) in datum_features.iteritems():
				self.empirical_counts[datum_label][feature] += cnt

	def get_log_probabilities(self, datum_features, weights):
		log_probs = Counter((label, sum((weights[label] * datum_features).itervalues())) for label in self.labels)
		log_probs.log_normalize()
		return log_probs

	last_vg_weights = None
	last_vg = (None, None)
	def value_and_gradient(self, weights, verbose=False):
		if weights == self.last_vg_weights:
			return self.last_vg
		objective = 0.0
		gradient = CounterMap()

		if verbose: print "Calculating log probabilities and objective..."

		# log_prob
		log_probs = list()
		for pos, (label, features) in enumerate(self.labeled_extracted_features):
			log_probs.append(get_log_probs(features, weights, self.labels))
			assert abs(sum(exp(log_probs[pos][label]) for label in self.labels) - 1.0) < 0.0001, "Not a distribution: P[any | features] = %f" % (sum(exp(log_probs[pos][label]) for label in self.labels))

		objective = -sum(log_prob[label] for (log_prob, (label,_)) in zip(log_probs, self.labeled_extracted_features))

		if verbose: print "Raw objective: %f" % objective

		if verbose: print "Calculating expected counts..."

		expected_counts = get_expected_counts(self.labeled_extracted_features, self.labels, log_probs, CounterMap())

		if verbose: print "Calculating gradient..."

		gradient = expected_counts - self.empirical_counts

		if verbose: print "Applying penalty"
		
		# Apply a penalty (e.g. smooth the results)
		if self.sigma:
			penalty = 0.0

			for label, feature_weights in gradient.iteritems():
				for feature in feature_weights:
					weight = weights[label][feature]
					penalty += weight**2
					gradient[label][feature] += (weight / (self.sigma**2))

			penalty /= 2 * self.sigma**2
			objective += penalty
			if verbose: print "Penalized objective: %f" % objective

		self.last_vg_weights = weights
		self.last_vg = (objective, gradient)
		return (objective, gradient)

	def value(self, weights, verbose=False):
		objective = 0.0

		if verbose: print "Calculating log probabilities and objective..."
		log_probs = list()
		for pos, (label, features) in enumerate(self.labeled_extracted_features):
			log_probs.append(get_log_probs(features, weights, self.labels))

		objective = -sum(log_probs[index][label] for (index, (label,_)) in enumerate(self.labeled_extracted_features))

		if verbose: print "Raw objective: %f" % objective

		if verbose: print "Applying penalty"

		if self.sigma:
			# Apply a penalty (e.g. smooth the results)
			penalty = sum(sum(weight**2 for weight in feature_weights.itervalues()) for feature_weights in weights.itervalues())
			penalty /= 2 * self.sigma**2
			objective += penalty

			if verbose: print "Penalized objective: %f" % objective

		return objective

class MaximumEntropyClassifier(object):
	labels = None
	features = None
	weights = None

	def __init__(self, labels=None, features=None):
		self.labels = labels
		self.features = features		

	def get_log_probabilities(self, datum_features):
		return get_log_probs(datum_features, self.weights, self.labels)
	
	def train_with_features(self, labeled_features, sigma=None, quiet=False):
		print "Optimizing weights..."
		weight_function = MaxEntWeightFunction(labeled_features, self.labels, self.features)
		weight_function.sigma = sigma

		print "Building initial dictionary..."
		initial_weights = CounterMap()

		print "Training on %d labelled features" % (len(labeled_features))

		print "Minimizing..."
		self.weights = Minimizer.minimize(weight_function, initial_weights, quiet=quiet)

	def train(self, labeled_data):
		self.labels, self.features = set(), set()

		print "Building features..."
		labeled_features = []
		for label, datum in labeled_data:
			self.labels.add(label)
			features = Counter()

			for feature in ngrams(datum, 1):
				features[feature] += 1.0
				self.features.add(feature)

			labeled_features.append((label, features))

		print "%d features" % len(self.features)
		print "%d labels" % len(self.labels)
			
		self.train_with_features(labeled_features)

	def label(self, datum):
		datum_features = Counter()
		for feature in ngrams(datum, 1):
			datum_features[feature] += 1.0

		log_probs = self.get_log_probabilities(datum_features)

		return log_probs.arg_max()
		
	def label_distribution(self, datum):
		datum_features = Counter()
		for feature in ngrams(datum, 1):
			datum_features[feature] += 1.0

		log_probs = self.get_log_probabilities(datum_features)

		return log_probs

def read_delimited_data(file_name):
	pairs = list()

	with open(file_name, "r") as delimited_file:
		for line in delimited_file.readlines():
			pair = line.rstrip().split("\t")
			pairs.append(pair)

	return pairs

def real_problem():
 	training_data = read_delimited_data("data/pnp-train.txt")
 	testing_data = read_delimited_data("data/pnp-test.txt")

 	classifier = MaximumEntropyClassifier()
 	classifier.train(training_data)

 	print "Correctly labeled %d of %d" % (sum(1 for (label, datum) in testing_data if classifier.label(datum) == label), len(testing_data))

	return classifier

def cnter(l):
	return Counter(izip(l, repeat(1.0, len(l))))

if __name__ == "__main__":
 	print "*** Maximum Entropy Classifier ***"

	real_problem()

