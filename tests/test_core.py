import unittest
from fractions import Fraction

from sturm_sequence import Polynomial, sturm_sequence, count_real_roots, isolate_real_roots


class TestPolynomial(unittest.TestCase):
    def test_construction_strips_leading_zeros(self):
        p = Polynomial([1, 2, 3, 0, 0])
        self.assertEqual(p.coeffs, [Fraction(1), Fraction(2), Fraction(3)])

    def test_zero_polynomial_is_single_zero(self):
        p = Polynomial([0, 0, 0])
        self.assertEqual(p.coeffs, [Fraction(0)])
        self.assertEqual(p.degree, -1)
        self.assertTrue(p.is_zero())

    def test_empty_coeffs_raises(self):
        with self.assertRaises(ValueError):
            Polynomial([])

    def test_call_evaluates_correctly(self):
        p = Polynomial([1, 2, 3])  # 3x^2 + 2x + 1
        self.assertEqual(p(0), Fraction(1))
        self.assertEqual(p(1), Fraction(6))
        self.assertEqual(p(Fraction(1, 2)), Fraction(11, 4))

    def test_derivative(self):
        p = Polynomial([5, 0, 0, 1])  # x^3 + 5
        d = p.derivative()
        self.assertEqual(d.coeffs, [Fraction(0), Fraction(0), Fraction(3)])

    def test_derivative_of_constant_is_zero(self):
        p = Polynomial([7])
        self.assertTrue(p.derivative().is_zero())

    def test_multiplication(self):
        a = Polynomial([1, 1])  # x + 1
        b = Polynomial([1, -1])  # x - 1
        prod = a * b
        self.assertEqual(prod.coeffs, [Fraction(1), Fraction(0), Fraction(-1)])

    def test_subtraction(self):
        a = Polynomial([1, 2, 3])
        b = Polynomial([0, 1, 1])
        self.assertEqual((a - b).coeffs, [Fraction(1), Fraction(1), Fraction(2)])

    def test_floordiv(self):
        a = Polynomial([-1, 0, 0, 0, 1])  # x^4 - 1
        b = Polynomial([1, 0, 1])  # x^2 + 1
        q = a.floordiv(b)
        self.assertEqual(q.coeffs, [Fraction(-1), Fraction(0), Fraction(1)])

    def test_mod(self):
        a = Polynomial([-1, 0, 0, 0, 1])
        b = Polynomial([1, 0, 1])
        r = a.mod(b)
        self.assertTrue(r.is_zero())

    def test_monic_part(self):
        p = Polynomial([2, 4, 6])  # 6x^2 + 4x + 2
        m = p.monic_part()
        self.assertEqual(m.coeffs, [Fraction(1, 3), Fraction(2, 3), Fraction(1)])


class TestSturmSequence(unittest.TestCase):
    def test_simple_linear(self):
        p = Polynomial([-1, 1])  # x - 1
        seq = sturm_sequence(p)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], p)
        self.assertEqual(seq[1], Polynomial([1]))

    def test_quadratic_with_two_roots(self):
        p = Polynomial([-1, 0, 1])  # x^2 - 1
        seq = sturm_sequence(p)
        # p0 = x^2 - 1, p1 = 2x, p2 = remainder of (x^2-1) mod (x) negated = -(-1) = 1
        self.assertEqual(seq[0], p)
        self.assertEqual(seq[1], Polynomial([0, 2]))
        self.assertEqual(seq[2], Polynomial([1]))

    def test_zero_polynomial_raises(self):
        with self.assertRaises(ValueError):
            sturm_sequence(Polynomial([0]))

    def test_constant_polynomial_raises(self):
        with self.assertRaises(ValueError):
            sturm_sequence(Polynomial([5]))


class TestCountRealRoots(unittest.TestCase):
    def test_x_squared_minus_one_has_two_real_roots(self):
        p = Polynomial([-1, 0, 1])
        self.assertEqual(count_real_roots(p), 2)

    def test_x_squared_plus_one_has_no_real_roots(self):
        p = Polynomial([1, 0, 1])
        self.assertEqual(count_real_roots(p), 0)

    def test_cubic_with_three_real_roots(self):
        # (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        p = Polynomial([-6, 11, -6, 1])
        self.assertEqual(count_real_roots(p), 3)

    def test_count_in_interval(self):
        p = Polynomial([-6, 11, -6, 1])  # roots at 1, 2, 3
        self.assertEqual(count_real_roots(p, Fraction(0), Fraction(5)), 3)
        self.assertEqual(count_real_roots(p, Fraction(3, 2), Fraction(5)), 2)
        self.assertEqual(count_real_roots(p, Fraction(0), Fraction(2)), 1)
        self.assertEqual(count_real_roots(p, Fraction(2), Fraction(3)), 0)

    def test_half_open_right_endpoint(self):
        # Root exactly at b=2 should NOT be counted.
        p = Polynomial([-6, 11, -6, 1])
        self.assertEqual(count_real_roots(p, Fraction(0), Fraction(2)), 1)
        # Root exactly at a=1 SHOULD be counted.
        self.assertEqual(count_real_roots(p, Fraction(1), Fraction(4)), 2)

    def test_repeated_root_counts_once(self):
        # (x-1)^2 = x^2 - 2x + 1
        p = Polynomial([1, -2, 1])
        self.assertEqual(count_real_roots(p), 1)


class TestIsolateRealRoots(unittest.TestCase):
    def test_isolate_three_distinct_roots(self):
        p = Polynomial([-6, 11, -6, 1])  # roots 1, 2, 3
        intervals = isolate_real_roots(p)
        self.assertEqual(len(intervals), 3)
        # Each interval must contain exactly one root.
        roots = [Fraction(1), Fraction(2), Fraction(3)]
        for (lo, hi), root in zip(intervals, roots):
            self.assertLess(lo, root)
            self.assertLess(root, hi)
        # Intervals are ordered and disjoint.
        for i in range(len(intervals) - 1):
            self.assertLessEqual(intervals[i][1], intervals[i + 1][0])

    def test_isolate_no_roots(self):
        p = Polynomial([1, 0, 1])  # x^2 + 1
        self.assertEqual(isolate_real_roots(p), [])

    def test_isolate_within_interval(self):
        p = Polynomial([-6, 11, -6, 1])  # roots 1, 2, 3
        intervals = isolate_real_roots(p, Fraction(3, 2), Fraction(5))
        self.assertEqual(len(intervals), 2)
        for lo, hi in intervals:
            self.assertGreaterEqual(lo, Fraction(3, 2))
            self.assertLess(hi, Fraction(5))

    def test_isolate_repeated_root(self):
        p = Polynomial([1, -2, 1])  # (x-1)^2
        intervals = isolate_real_roots(p)
        self.assertEqual(len(intervals), 1)
        lo, hi = intervals[0]
        self.assertLess(lo, Fraction(1))
        self.assertLess(Fraction(1), hi)

    def test_isolate_rational_coefficients(self):
        # roots at 1/2 and 3/2: (2x-1)(2x-3) = 4x^2 - 8x + 3
        p = Polynomial([3, -8, 4])
        intervals = isolate_real_roots(p)
        self.assertEqual(len(intervals), 2)
        roots = [Fraction(1, 2), Fraction(3, 2)]
        for (lo, hi), root in zip(intervals, roots):
            self.assertLess(lo, root)
            self.assertLess(root, hi)

    def test_isolate_high_degree(self):
        # (x+2)(x+1)x(x-1)(x-2) = x^5 - 5x^3 + 4x
        p = Polynomial([0, 4, 0, -5, 0, 1])
        intervals = isolate_real_roots(p)
        self.assertEqual(len(intervals), 5)
        roots = [Fraction(-2), Fraction(-1), Fraction(0), Fraction(1), Fraction(2)]
        for (lo, hi), root in zip(intervals, roots):
            self.assertLess(lo, root)
            self.assertLess(root, hi)


if __name__ == "__main__":
    unittest.main()
