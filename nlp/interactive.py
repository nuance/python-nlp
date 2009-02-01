import maximumentropy
from counter import Counter

classifier = maximumentropy.toy_problem()

datum_features = Counter()

for feature in classifier.extract_features("Test datum"):
	datum_features[feature] += 1.0

sums = Counter()

for label in classifier.labels:
	for k, v in datum_features.iteritems():
		print "feature %s (%s): %.2f * %.2f = %.2f" % (k, label, v, classifier.weights[label][k], v * classifier.weights[label][k])
		sums[label] += v * classifier.weights[label][k]

	print "Label %s has score %.2f" % (label, sums[label])


