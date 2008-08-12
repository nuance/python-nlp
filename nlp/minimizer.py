import function
from itertools import izip

class Minimizer:
	
	max_iterations = 1000
	sqr_convergence = 0.000000000001
	step = 0.01
	verbose = False

	@classmethod
	def minimize(cls, function, start):
		converged = False
		iteration = 0
		point = start
		
		while not converged:
			if cls.verbose: print "*** Starting gradient descent iteration %d ***" % iteration
			
			(value, gradient) = function.value_and_gradient(point)
			
			next_point = [coord - cls.step * partial for (coord, partial) in izip(point, gradient)]
			iteration += 1
			
			if sum((start - stop)**2 for (start, stop) in izip(point, next_point)) < cls.sqr_convergence:
				converged = True
			elif iteration > cls.max_iterations:
				converged = True
			
			point = next_point
		
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