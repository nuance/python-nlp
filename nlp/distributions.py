from counter import counter_map
from math import log, pi

class Gaussian(object):
	@classmethod
	def log_prob(cls, point, mean, precision, debug=False):
		# - (x - \mu)^2 / 2 \sigma ^2 => - 0.5 * precision * (x - \mu)^2
		result = -0.5 * precision * (point - mean) ** 2
		# (result) - log(\sigma * sqrt(2 * pi))
		# => - (log(\sigma) + 0.5 * log(2 * pi))
		# log(\sigma) => log(1 / sqrt(precision)) => 1 - 0.5 * log(precision)
		# => - 1 + 0.5 * log(precision) - 0.5 * log(2 * pi))
		result -= 1
		result += 0.5 * counter_map(precision, lambda pr: log(pr) if pr else float("-inf"))
		result -= 0.5 * log(2 * pi)

		if debug or any(v > 0.0 for v in result.itervalues()):
			print "GaussianDistribution.log_prob"
			print " - mean:      ", mean
			print " - precision: ", precision
			print " - point:     ", point
			print "  [1]  -0.5 * precision =", -0.5 * precision
			print "  [2]  (point - mean) ** 2 =", (point - mean) ** 2
			print "  [1] * [2] =", (-0.5 * precision * (point - mean) ** 2)
			print "  [3]  0.5 * log(tau) =", 0.5 * counter_map(precision, lambda pr: log(pr) if pr else float("-inf"))
			print "  [4]  -0.5 * log(2*pi) =", - 0.5 * log(2 * pi)

			print "= %s" % result
			raise Exception()

		return result

