from copy import copy

from countermap import CounterMap
from features import ngrams

class NaiveBayesClassifier:
	def train(self, labeled_data):
		self.feature_distribution = CounterMap()
		labels = set()

		for label, datum in labeled_data:
			labels.add(label)
			for feature in ngrams(datum, 3)
				self.feature_distribution[feature][label] += 1

		for feature in self.feature_distribution.iterkeys():
			self.feature_distribution[feature].default = 0.01

		self.feature_distribution.normalize()
		self.feature_distribution.log()

	def label_distribution(self, datum):
		distribution = None

		for feature in ngrams(datum, 3):
			if distribution:
				distribution += self.feature_distribution[feature]
			else:
				distribution = copy(self.feature_distribution[feature])

		distribution.log_normalize()

		return distribution

	def label(self, datum):
		distribution = None

		for feature in ngrams(datum, 3):
			if distribution:
				distribution += self.feature_distribution[feature]
			else:
				distribution = copy(self.feature_distribution[feature])

		return distribution.arg_max()

def read_delimited_data(file_name):
	delimited_file = open(file_name, "r")
	pairs = list()

	for line in delimited_file.readlines():
		pair = line.rstrip().split("\t")
		pair.reverse()
		pairs.append(pair)

	return pairs

if __name__ == "__main__":
	print "*** Naive Bayes Classifier ***"
	training_data = read_delimited_data("data/pnp-train.txt")
	testing_data = read_delimited_data("data/pnp-test.txt")

	classifier = NaiveBayesClassifier()
	classifier.train((label, datum) for datum, label in training_data)

	print "Correctly labeled %d of %d" % (sum(1 for (datum, label) in testing_data if classifier.label(datum) == label), len(testing_data))
