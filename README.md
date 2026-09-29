# sturm_sequence

Counts and isolates real roots of a polynomial with rational coefficients using Sturm's theorem. All arithmetic is exact (no floating point), so sign tests never suffer from rounding.

## Usage

```python
from fractions import Fraction
from sturm_sequence import Polynomial, count_real_roots, isolate_real_roots

# (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
p = Polynomial([-6, 11, -6, 1])

print(count_real_roots(p))            # 3
print(count_real_roots(p, Fraction(0), Fraction(2)))  # 1

for lo, hi in isolate_real_roots(p):
    print(float(lo), float(hi))
# e.g. 0.0 1.5
#      1.5 2.5  (contains root 2)
#      2.5 4.0  (contains root 3)
```

## Why this exists

Numerical root-finders (Newton's method, companion-matrix eigenvalues) give approximate answers and can miss close roots or report spurious ones. Sturm's theorem gives an exact count of distinct real roots in any interval and, combined with bisection, isolates each root to an arbitrary-precision rational interval. The trade-off is speed: exact rational arithmetic is slower than floating point, and bisection narrows intervals linearly. For polynomials of degree up to a few dozen this is rarely a problem; for degree in the hundreds it will be.

## Edge cases

- **Repeated roots are counted once.** Sturm's theorem counts distinct real roots. `(x-1)^2` has one distinct real root; `isolate_real_roots` returns a single interval for it.
- **Intervals are half-open on the right** (`[a, b)`). A root exactly at `b` is not counted by `count_real_roots`; a root exactly at `a` is. This matches the standard statement of Sturm's theorem and prevents double-counting during bisection.
- **The zero polynomial and constants raise `ValueError`.** Sturm's theorem requires a non-constant polynomial.
- When no interval is given, `isolate_real_roots` uses a Cauchy bound to find a finite starting interval that contains all real roots.

## Exported names

- `Polynomial` — univariate polynomial with rational coefficients. Constructor takes a list of coefficients little-endian (`coeffs[i]` is the coefficient of `x^i`).
- `sturm_sequence(poly)` — returns the canonical Sturm sequence as a list of `Polynomial`.
- `count_real_roots(poly, a=None, b=None)` — number of distinct real roots in `[a, b)`. `None` means ±∞.
- `isolate_real_roots(poly, a=None, b=None)` — list of disjoint open `(lo, hi)` intervals, each containing exactly one distinct real root. Endpoints are `fractions.Fraction`.
