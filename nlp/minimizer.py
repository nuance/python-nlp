from copy import copy
from itertools import izip
from time import time

class Minimizer(object):
	min_iterations = 0
	max_iterations = 25
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
	def minimize(cls, function, start_map, verbose=False, quiet=False):
		converged = False
		iteration = 0
		point = start_map
		last_time = time()

		history = list()

		derivative_delta = None
		point_delta = None

		while not converged:
			(value, gradient) = function.value_and_gradient(point)
			if verbose: print "Found value and gradient"

			# Calculate inverse hessian scaling
			hessian_scale = 1.0
			if derivative_delta:
				hessian_scale = derivative_delta.inner_product(point_delta) / derivative_delta.inner_product(derivative_delta)
			if verbose: print "Found hessian scaling: %f" % hessian_scale

			# Find and invert direction
			direction = type(start_map)() - cls.__implicit_multiply(hessian_scale, gradient, history)
			if verbose: print "Found Direction"

			# Line search in the direction found
			if iteration == 0:
				# Breaking out from the first value requires smaller steps in most cases,
				# so don't overdo the step size
				next_point = cls.__line_minimize(function, point, direction, step_size_mult=0.01)
			else:
				next_point = cls.__line_minimize(function, point, direction, step_size_mult=0.5)

			if verbose: print "Line minimization done"

			# This function call should be cached for the next iteration
			(next_value, next_gradient) = function.value_and_gradient(next_point)

			converged = iteration > cls.min_iterations and abs((next_value - value) / ((next_value + value + cls.epsilon) / 2.0)) < cls.tolerance

			converged = converged or iteration >= cls.max_iterations

			history.append((next_point - point, next_gradient - gradient))
			point = next_point
			iteration += 1

			if not quiet: print "*** Minimizer finished iteration %d with objective %f" % (iteration, next_value)

		return point
