import function
from copy import copy
from itertools import izip
from time import time

class Minimizer:
	max_iterations = 0
	sqr_convergence = 0.000000000001
	step = 0.01
	verbose = True

	@classmethod
	def minimize(cls, function, start):
		converged = False
		iteration = 0
		point = start
		last_time = time()
		
		while not converged:
			(value, gradient) = function.value_and_gradient(point)
			
			next_point = [coord - cls.step * partial for (coord, partial) in izip(point, gradient)]
			iteration += 1
			
			if sum((start - stop)**2 for (start, stop) in izip(point, next_point)) < cls.sqr_convergence:
				converged = True
			elif iteration > cls.max_iterations:
				converged = True
			
			point = next_point

			if cls.verbose and time()-last_time > 1:
				print "*** Finished gradient descent iteration %d (objective: %f)***" % (iteration, value)
				last_time = time()

		return point

	@classmethod
	def minimize_map(cls, function, start_map):
		converged = False
		iteration = 0
		point = start_map
		last_time = time()

		subkeys = None
		
		while not converged:
			if subkeys:
				for key in ['person', 'movie']:
					print "Point (%s): %s" % (key, [point[key][subkey] for subkey in subkeys])

			tup = function.value_and_gradient(point)
			(value, gradient) = tup

			next_point = type(point)()
#			for (key, counter) in point.iteritems():
#				next_point[key] = type(counter)()
#				for (sub_key, val) in counter.iteritems():
#					next_point[key][sub_key] = val
			change = 0.0

			for (key, partials) in gradient.iteritems():
				deltas = partials * cls.step
				next_point[key] = point[key] - deltas
				change += sum(delta**2 for delta in deltas.itervalues())
			iteration += 1
			
			if change < cls.sqr_convergence:
				converged = True
			elif iteration > cls.max_iterations:
				converged = True

			for key in ['person', 'movie']:
				partials = gradient[key]
				print "Key: %s" % key
				subkeys = partials.keys()[0:5]
				print subkeys
				print "Partials: %s" % [partials[subkey] for subkey in subkeys]
				print "Deltas: %s" % [deltas[subkey] for subkey in subkeys]
				print "Point: %s" % [point[key][subkey] for subkey in subkeys]
				print "Next point: %s" % [next_point[key][subkey] for subkey in subkeys]
			
			point = next_point

			if cls.verbose and time()-last_time > 1 or iteration > cls.max_iterations:
				print "*** Finished gradient descent iteration %d (objective: %f)***" % (iteration, value)
				last_time = time()
		
		return point

def test():

	class TwoDimPolynomial(function.Function):
		"""
		2x^2 - 10x + 27
		Minimum at 4x-10 = 0: x = 2.5
		"""
		def value_and_gradient(self, point):
			value = 2 * (point[0]-5)**2 + 2 # 2(x-5)^2+2 => 2x^2 - 10x + 27
			gradient = (4 * point[0] - 10,)
			return (value, gradient)

		def value(self, point):
			return 2 * (point[0]-5)**2 + 2

	Minimizer.max_iterations = 1000

	twodimfunc = TwoDimPolynomial()
	min_point = Minimizer.minimize(twodimfunc, (0,))
	assert sum((min_coord - true_coord)**2 for (min_coord, true_coord) in izip(min_point, (2.5,))) < 0.001

	class ThreeDimPolynomial(function.Function):
		"""
		2x^2 - y - 2y^2 + x = z
		Minimum at (4x+1==0, 4y-1==0)
		"""

		def value_and_gradient(self, point):
			(x, y) = point
			value = 2*x**2 - y - 2*y**2 + x
			gradient = (4*x+1, 4*y-1)
			return (value, gradient)

		def value(self, point):
			(x, y) = point
			return 2*x**2 - y + 3*y**3 - 2*y**2 + x

	threedimfunc = ThreeDimPolynomial()
	min_point = Minimizer.minimize(threedimfunc, (0,0))
	assert sum((min_coord - true_coord)**2 for (min_coord, true_coord) in izip(min_point, (-0.25,0.25))) < 0.001

if __name__ == "__main__":
	test()
