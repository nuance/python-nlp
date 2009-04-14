include "math.pxi"

DEF ROOT2 = 1.4142135623730951

def g_cdf(double x, double mean, double variance):
	return 0.5 + 0.5 * erf((x - mean) / (variance * ROOT2))

def g_log_cdf(double x, double mean, double variance):
	return log(0.5 + 0.5 * erf((x-mean) / (variance * ROOT2)))
