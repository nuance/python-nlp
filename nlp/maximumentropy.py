import sys
from math import exp, log

# c modules
from nlp import counter as Counter
from maxent import get_log_probabilities as get_log_probs

# python modules
from countermap import CounterMap
#from counter import Counter
from function import Function
from minimizer import Minimizer
from itertools import izip, chain, repeat

def slow_log_probs(datum_features, weights, labels):
	log_probs = Counter((label, sum((weights[label] * datum_features).itervalues())) for label in labels)
	log_probs.log_normalize()
	return log_probs

#get_log_probs = slow_log_probs

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

	def value_and_gradient(self, weights, verbose=False):
		objective = 0.0
		gradient = CounterMap()

		if verbose: print "Calculating log probabilities and objective..."
		log_probs = list()
		for pos, (label, features) in enumerate(self.labeled_extracted_features):
			log_probs.append(get_log_probs(features, weights, self.labels))
			assert abs(sum(exp(log_probs[pos][label]) for label in self.labels) - 1.0) < 0.0001, "Not a distribution: P[any | features] = %f" % (sum(exp(log_probs[pos][label]) for label in self.labels))

		objective = -sum(log_probs[index][label] for (index, (label,_)) in enumerate(self.labeled_extracted_features))

		if verbose: print "Raw objective: %f" % objective

		if verbose: print "Calculating expected counts..."

		expected_counts = CounterMap()

		for (index, (_, datum_features)) in enumerate(self.labeled_extracted_features):
			for (feature, cnt) in datum_features.iteritems():
				for label in self.labels:
					expected_counts[label][feature] += exp(log_probs[index][label]) * cnt

		if verbose: print "Calculating gradient..."

		gradient = expected_counts - self.empirical_counts

		if verbose: print "Applying penalty"
		
		# Apply a penalty (e.g. smooth the results)
		penalty = 0.0

		for label in self.labels:
			for feature in self.features:
				weight = weights[feature][label]
				penalty += weight**2
				gradient[label][feature] += (weight / (self.sigma**2))

		penalty /= 2 * self.sigma**2
		objective += penalty
		if verbose: print "Penalized objective: %f" % objective

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
		
		# Apply a penalty (e.g. smooth the results)
		penalty = 0.0

		for label in self.labels:
			for feature in self.features:
				weight = weights[feature][label]
				penalty += weight**2

		penalty /= 2 * self.sigma**2
		objective += penalty
		if verbose: print "Penalized objective: %f" % objective

		return objective

class MaximumEntropyClassifier:
	labels = None
	features = None
	weights = None

	def get_log_probabilities(self, datum_features):
		return get_log_probs(datum_features, self.weights, self.labels)
	
	def extract_features(self, datum):
#		for word in datum.split():
#			yield word
		last_last_char = ''
		last_char = ''
		for char in datum:
			yield char
#			yield last_char+char
#			yield last_last_char + last_char + char
			last_last_char = last_char
			last_char = char

	def train_with_features(self, labeled_features):
		print "Optimizing weights..."
		weight_function = MaxEntWeightFunction(labeled_features, self.labels, self.features)

		print "Building initial dictionary..."
		initial_weights = CounterMap()

		print "Labels: %s" % self.labels

		print "Minimizing..."
		self.weights = Minimizer.minimize(weight_function, initial_weights)

	def train(self, labeled_data):
		print "Building label set"
		self.labels = set(label for _,label in labeled_data)

		self.features = set()

		print "Labeling data..."
		labeled_features = []
		for (datum, label) in labeled_data:
			features = Counter()
			for feature in self.extract_features(datum):
				features[feature] += 1.0
				self.features.add(feature)
			labeled_features.append((label, features))

		print "%d features" % len(self.features)
			
		self.train_with_features(labeled_features)

	def label(self, datum):
		datum_features = Counter()
		for feature in self.extract_features(datum):
			datum_features[feature] += 1.0

		log_probs = self.get_log_probabilities(datum_features)

		return log_probs.arg_max()

def read_delimited_data(file_name):
	delimited_file = open(file_name, "r")
	pairs = list()

	for line in delimited_file.readlines():
		pair = line.rstrip().split("\t")
		pair.reverse()
		pairs.append(pair)

	return pairs

def real_problem():
 	training_data = read_delimited_data("data/pnp-train.txt")
 	testing_data = read_delimited_data("data/pnp-test.txt")

 	classifier = MaximumEntropyClassifier()
 	classifier.train(training_data)

 	print "Correctly labeled %d of %d" % (sum(1 for (datum, label) in testing_data if classifier.label(datum) == label), len(testing_data))

	return classifier

def cnter(l):
	return Counter(izip(l, repeat(1.0, len(l))))

def toy_problem():
	training_data = (('cat', cnter(('fuzzy', 'claws', 'small'))),
					 ('bear', cnter(('fuzzy', 'claws', 'big'))),
					 ('cat', cnter(('claws', 'medium'))))
	test_data = (('cat', cnter(('claws', 'small'))),
				 ('bear', cnter(('fuzzy',))))

	classifier = MaximumEntropyClassifier()
	classifier.labels = set(('cat', 'bear'))
	classifier.features = set(('fuzzy', 'claws', 'small', 'medium', 'big'))
	classifier.train_with_features(training_data)

	print "Weights: %s" % classifier.weights
	for test_datum in test_data:
		log_probs = classifier.get_log_probabilities(test_datum[1])
		for label in ['cat', 'bear']:
			print "P[%s | %s] = %f" % (label, test_datum[1], exp(log_probs[label]))

if __name__ == "__main__":
 	print "*** Maximum Entropy Classifier ***"

	if 'toy' in sys.argv:
		toy_problem()
	else:
		real_problem()

