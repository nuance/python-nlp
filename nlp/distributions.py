from math import log

import numpy
from scipy import linalg
from scipy.stats import chi2, norm

from countermap import CounterMap
from lib import mvncdf
from future_math import gaussian_cdf

class Gaussian(object):
	"""multivariate gaussian with independent components
	"""
	@classmethod
	def prob(cls, point, mean, precision, debug=False, discretization=0.1):
		prob = 1.0

		for key, pt in point.iteritems():
			prob *= abs(gaussian_cdf(pt+discretization, mean[key], precision[key])
						- gaussian_cdf(pt-discretization, mean[key], precision[key]))

		return prob

	@classmethod
	def log_prob(cls, point, mean, precision, debug=False, discretization=0.1):
		log_prob = 0.0

		for key, pt in point.iteritems():
			prob = abs(gaussian_cdf(pt+discretization, mean[key], precision[key])
					   - gaussian_cdf(pt-discretization, mean[key], precision[key]))
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


class InverseWishart(object):
	"""rejection-sampler for estimating probs
	"""
	@classmethod
	def log_prob(cls, matrix, degree_of_freedom, inverse_scale):
		return log(cls.prob(matrix, degree_of_freedom, inverse_scale))

	@classmethod
	def prob(cls, matrix, degree_of_freedom, inverse_scale):
		keys, inv_scale_matrix = inverse_scale.matrix()
		scale = inv_scale_matrix.inv()

		samples = 1000.0
		hits = 0.0

		for _ in xrange(samples):
			m = Wishart.sample(degree_of_freedom, scale)

			if sum(matrix - m) < discretization:
				hits += 1.0

		return hits / samples


class Wishart(object):
	@classmethod
	def sample(cls, degree_of_freedom, scale):
		# From wikipedia:
		# 1) sample a[i][i] from a chi-square distribtion of \chi^2_{dof - i - 1}
		# 2) draw a[i][j | j < i] from a normal(0, 1) distribution
		# 3) compute cholesky decomposition of the scale matrix, L
		# 4) X, sampled from wishart, = L A A^T L^T
		# sample these in bulk
		n = scale.shape[0]
		a_lower = norm.rvs(size=(n * (n - 1) / 2))

		def fill(x, y):
			if x == y:
				return chi2.rvs(degree_of_freedom - x - 1)[0]
			elif x < y:
				return a_lower.pop()
			else:
				return 0

		# construct A
		a = numpy.fromfunction(fill, scale.shape)
		l = linalg.cholesky(scale)

		return l * a * a.T * l.T

