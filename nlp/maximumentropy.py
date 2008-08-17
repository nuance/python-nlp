from math import exp, log

from countermap import CounterMap
from nlp import counter
from function import Function
from minimizer import Minimizer

class MaxEntWeightFunction(Function):
	sigma = 1.0
	labels = None
	features = None

	def __init__(self, labeled_extracted_features, labels, features):
		self.labeled_extracted_features = labeled_extracted_features
		self.labels = labels
		self.features = features

	def get_log_probabilities(self, datum_features, weights):
		log_probs = counter((label, self.weights[label] * datum_features) for label in self.labels)
		log_probs.log_normalize()
		return log_probs

	def value_and_gradient(self, weights):
		objective = 0.0
		gradient = CounterMap()

		log_probs = (self.get_log_probabilities(labeled_datum, weights) for labeled_datum in self.labeled_extracted_features)
		objective = -sum(log_probs[index][label] for (index, (label,_)) in enumerate(self.labeled_extracted_features))

		for label in self.labels:
			for feature in self.features:
				empirical_count = 0.0
				expected_count = 0.0

				for (index, (datum_label, datum_features)) in enumerate(self.labeled_extracted_features):
					if feature in datum_features:
						if datum_label == label:
							empirical_count += datum_features[feature]
						expected_count += exp(log_probs[index][label]) * datum_features[feature]

					gradient[label][feature] = expected_count - empirical_count

		# Apply a penalty (e.g. smooth the results)
		penalty = 0.0

		for label in self.labels:
			for feature in self.features:
				weight = weights[feature][label]
				penalty += weight**2
				gradient[dim] += (weight / (self.sigma**2))

		penalty /= 2 * self.sigma**2
		objective += penalty

		return (objective, gradient)

class MaximumEntropyClassifier:
	def extract_features(self, datum):
		# for word in datum.split():
		# yield word
		last_last_char = ''
		last_char = ''
		for char in datum:
			yield char
			yield last_char+char
			yield last_last_char + last_char + char
			last_last_char = last_char
			last_char = char

	def train(self, labeled_data):
		self.labels = set(label for _,label in labeled_data)

		print "Labeling data..."
		labeled_features = [(label, counter(self.extract_features(datum))) for (datum, label) in labeled_data]

		print "Optimizing weights..."
		weight_function = MaxEntWeightFunction(labeled_features, self.labels)
		self.weights = Minimizer.minimize(weight_function, CounterMap())

	def label(self, datum):
		datum_features = counter(self.extract_features(datum))
		log_probs = counter((label, self.weights[label] * datum_features) for label in self.labels)

		return log_probs.arg_max()

def read_delimited_data(file_name):
	delimited_file = open(file_name, "r")
	pairs = list()

	for line in delimited_file.readlines():
		pair = line.rstrip().split("\t")
		pair.reverse()
		pairs.append(pair)

	return pairs

if __name__ == "__main__":
	print "*** Maximum Entropy Classifier ***"
	training_data = read_delimited_data("data/pnp-train.txt")
	testing_data = read_delimited_data("data/pnp-test.txt")

	classifier = MaximumEntropyClassifier()
	classifier.train(training_data)

	print "Correctly labeled %d of %d" % (sum(1 for (datum, label) in testing_data if classifier.label(datum) == label), len(testing_data))
