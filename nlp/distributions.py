from math import log

from future_math import g_cdf

class Gaussian(object):
	@classmethod
	def prob(cls, point, mean, precision, debug=False, discretization=0.1):
		prob = 1.0

		for key, pt in point.iteritems():
			prob *= abs(g_cdf(pt+discretization, mean[key], precision[key])
						- g_cdf(pt-discretization, mean[key], precision[key]))

		return prob

	@classmethod
	def log_prob(cls, point, mean, precision, debug=False, discretization=0.1):
		log_prob = 0.0

		if debug: print "Looking at range %s => %s" % (point-discretization, point+discretization)
		if debug: print "Mean: %s" % mean
		if debug: print "Precision: %s" % precision
		for key, pt in point.iteritems():
			prob = abs(g_cdf(pt+discretization, mean[key], precision[key])
					   - g_cdf(pt-discretization, mean[key], precision[key]))
			if debug: print "prob: %f" % prob
			log_prob += log(prob) if prob else float("-inf")
		if debug: print "log_prob: %f" % log_prob

		return log_prob
