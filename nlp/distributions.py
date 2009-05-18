from math import log

from lib import mvncdf
from future_math import g_cdf

class Gaussian(object):
	"""multivariate gaussian with independent components
	"""
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

		for key, pt in point.iteritems():
			prob = abs(g_cdf(pt+discretization, mean[key], precision[key])
					   - g_cdf(pt-discretization, mean[key], precision[key]))
			log_prob += log(prob) if prob else float("-inf")

		return log_prob

class MultivariateGaussian(object):
	"""like Gaussian, but allows dependent components
	"""
	@classmethod
	def prob(cls, point, mean, covariance_matrix, debug=False, discretization=0.1):
		keys, matrix = covariance_matrix.matrix()

		lower = [point[key] - discretization for key in keys]
		upper = [point[key] + discretization for key in keys]
		mean = [mean[key] for key in keys]

		return mvncdf.mvnormcdf(lower, upper, mean, matrix)

	@classmethod
	def log_prob(cls, point, mean, covariance_matrix, debug=False, discretization=0.1):
		return log(cls.prob(point, mean, covariance_matrix, debug=debug, discretization=discretization))

