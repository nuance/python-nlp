from countermap import CounterMap
from nlp import counter

class NaiveBayesClassifier:
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
		self.feature_distribution = CounterMap()
		labels = set()

		for (datum, label) in labeled_data:
			labels.add(label)
			for feature in self.extract_features(datum):
				self.feature_distribution[feature][label] += 1

		for feature in self.feature_distribution.iterkeys():
			for label in labels:
				if self.feature_distribution[feature][label] == 0.0:
					self.feature_distribution[feature][label] = 0.01

		self.feature_distribution.normalize()

	def label(self, datum):
		label_distribution = counter()

		for feature in self.extract_features(datum):
			label_distribution *= self.feature_distribution[feature]

		return label_distribution.arg_max()

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
	classifier.train(training_data)

	print "Correctly labeled %d of %d" % (sum(1 for (datum, label) in testing_data if classifier.label(datum) == label), len(testing_data))
