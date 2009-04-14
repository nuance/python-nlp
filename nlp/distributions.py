from math import log, pi

from counter import counter_map
from future_math import g_cdf, g_log_cdf

class Gaussian(object):
	@classmethod
	def prob(cls, point, mean, precision, debug=False, discretization=0.00001):
		prob = 1.0

		for key, pt in point.iteritems():
			prob *= g_cdf(pt+discretization, mean[key], precision[key]) - g_cdf(pt-discretization, mean[key], precision[key])

		return prob

	@classmethod
	def log_prob(cls, point, mean, precision, debug=False, discretization=0.00001):
		prob = 0.0

		for key, pt in point.iteritems():
			prob += g_log_cdf(pt+discretization, mean[key], precision[key]) - g_cdf(pt-discretization, mean[key], precision[key])

		return prob
