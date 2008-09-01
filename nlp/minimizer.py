import function
from copy import copy
from itertools import izip
from time import time

class Minimizer:
	min_iterations = 0
	max_iterations = 5
	epsilon = 1e-10
	tolerance = 1e-4
	verbose = True

	@classmethod
	def __line_minimize(cls, function, start, direction, step_size_mult=0.9, verbose=False):
		step_size = 1

		(value, gradient) = function.value_and_gradient(start)
		derivative = direction.inner_product(gradient)

		guess = None
		guess_value = 0.0
		done = False

		if verbose: print "Starting with step size %f" % step_size
		
		while True:
			guess = start + direction * step_size
			guess_value = function.value(guess)
			sufficient_decrease_value = value + cls.tolerance * derivative * step_size

			if sufficient_decrease_value >= guess_value:
				return guess

			step_size *= step_size_mult
			if step_size < cls.epsilon:
				if verbose: print "Line searcher underflow"
				return start

			if verbose: print "Retrying with step size %f" % step_size

		assert False, "Line searcher should have returned by now!"


	@classmethod
	def __implicit_multiply(cls, scale, gradient, delta_history, verbose=False):
		rho = list()
		alpha = list()
		right = type(gradient)()

		for key, counter in gradient.iteritems():
			right[key] = copy(counter)

		for (point_delta, derivative_delta) in reversed(delta_history):
			rho.append(point_delta.inner_product(derivative_delta))
			if rho[-1] == 0.0:
				raise Exception("Curvature problem")
			alpha.append(point_delta.inner_product(right) / rho[-1])
			right += derivative_delta * (-alpha[-1])

		if verbose: print "Right: %s" % repr(right)
		if verbose: print "Scale: %f" % scale

		alpha.reverse()
		rho.reverse()
		left = right * scale

		for alpha, rho, (point_delta, derivative_delta) in izip(alpha, rho, delta_history):
			left += point_delta * (alpha - derivative_delta.inner_product(left) / rho)

		if verbose: print "Left: %s" % repr(left)

		return left

	@classmethod
	def minimize(cls, function, start_map, verbose=False):
		converged = False
		iteration = 0
		point = start_map
		last_time = time()

		history = list()

		derivative_delta = None
		point_delta = None

		while not converged:
			(value, gradient) = function.value_and_gradient(point)
			if verbose: print "Value: %s, Gradient: %s" % (value, gradient)

			# Calculate inverse hessian scaling
			hessian_scale = 1.0
			if derivative_delta:
				hessian_scale = derivative_delta.inner_product(point_delta) / derivative_delta.inner_product(derivative_delta)
			if verbose: print "Found hessian scaling: %f" % hessian_scale

			# Find and invert direction
			direction = type(start_map)() - cls.__implicit_multiply(hessian_scale, gradient, history)
			if verbose: print "Direction: %s" % repr(direction)

			# Line search in the direction found
			if iteration == 0:
				next_point = cls.__line_minimize(function, point, direction, step_size_mult=0.01)
			else:
				next_point = cls.__line_minimize(function, point, direction, step_size_mult=0.5)

			# This function call should be cached for the next iteration
			(next_value, next_gradient) = function.value_and_gradient(next_point)

			converged = iteration > cls.min_iterations and abs((next_value - value) / ((next_value + value + cls.epsilon) / 2.0)) < cls.tolerance

			converged = converged or iteration >= cls.max_iterations

			history.append((next_point - point, next_gradient - gradient))
			point = next_point
			iteration += 1

			print "*** Minimizer finished iteration %d with objective %f" % (iteration, next_value)

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
