from itertools import izip
import unittest

from minimizer import Minimizer
from counter import Counter
from function import Function

class MinimizerTest(unittest.TestCase):
	def test_two_dim_polynomial(self):
		class TwoDimPolynomial(Function):
			"""
			2x^2 - 10x + 27
			Minimum at 4x-10 = 0: x = 2.5
			"""
			def value_and_gradient(self, point):
				value = 2 * (point['y']-5)**2 + 2 # 2(x-5)^2+2 => 2x^2 - 10x + 27
				gradient = Counter()
				gradient['y'] = 4 * point['y'] - 10
				return (value, gradient)

			def value(self, point):
				return 2 * (point['y']-5)**2 + 2

		Minimizer.max_iterations = 1000

		twodimfunc = TwoDimPolynomial()
		start = Counter()
		start['y'] = 0.0
		min_point = Minimizer.minimize(twodimfunc, start, quiet=True)

		self.assertAlmostEqual(min_point['y'], 2.5, 3)

	def test_three_dim_polynomial(self):
		class ThreeDimPolynomial(Function):
			"""
			2x^2 - y - 2y^2 + x = z
			Minimum at (4x+1==0, 4y-1==0)
			"""

			def value_and_gradient(self, point):
				x, y = point['x'], point['y']
				value = 2*x**2 - y - 2*y**2 + x
				gradient = Counter({'x' : 4*x+1, 'y' :  4*y-1})
				return (value, gradient)

			def value(self, point):
				x, y = point['x'], point['y']
				return 2*x**2 - y + 3*y**3 - 2*y**2 + x

		threedimfunc = ThreeDimPolynomial()

		start = Counter()
		start['x'] = 0
		start['y'] = 0

		min_point = Minimizer.minimize(threedimfunc, start, quiet=True)

		self.assertAlmostEqual(min_point['x'], -0.25, 3)
		self.assertAlmostEqual(min_point['y'], 0.25, 3)

if __name__ == "__main__":
	unittest.main()

