"""Exact real-root counting and isolation for polynomials with rational coefficients.

All arithmetic is performed with :class:`fractions.Fraction` so that sign tests
are exact.  No floating point is used anywhere in the computation of the Sturm
sequence or in the bisection that isolates roots — the only place floats appear
is as the final, human-readable endpoints of an isolating interval, and even
there the underlying value remains a ``Fraction`` until it is rendered.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Polynomial",
    "sturm_sequence",
    "count_real_roots",
    "isolate_real_roots",
]


class Polynomial:
    """A univariate polynomial with rational coefficients.

    Coefficients are stored little-endian: ``coeffs[i]`` is the coefficient of
    ``x**i``.  A polynomial is never empty; the zero polynomial is represented
    as ``Polynomial([0])`` and has degree ``-1`` by convention so that the
    degree of a non-zero constant is ``0``.
    """

    __slots__ = ("coeffs",)

    def __init__(self, coeffs: Sequence):
        if not coeffs:
            raise ValueError("Polynomial requires at least one coefficient")
        normalized = [Fraction(c) for c in coeffs]
        # Strip leading zeros but always keep at least one coefficient so that
        # the zero polynomial is ``[0]`` rather than ``[]``.
        while len(normalized) > 1 and normalized[-1] == 0:
            normalized.pop()
        self.coeffs = normalized

    @property
    def degree(self) -> int:
        """Degree of the polynomial; ``-1`` for the zero polynomial."""
        if len(self.coeffs) == 1 and self.coeffs[0] == 0:
            return -1
        return len(self.coeffs) - 1

    def is_zero(self) -> bool:
        return self.degree == -1

    def __call__(self, x) -> Fraction:
        x = Fraction(x)
        # Horner's method, evaluated from the top down.
        result = Fraction(0)
        for c in reversed(self.coeffs):
            result = result * x + c
        return result

    def __mul__(self, other: "Polynomial") -> "Polynomial":
        if self.is_zero() or other.is_zero():
            return Polynomial([0])
        n, m = self.degree, other.degree
        out = [Fraction(0)] * (n + m + 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                out[i + j] += a * b
        return Polynomial(out)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        n = max(len(self.coeffs), len(other.coeffs))
        out = []
        for i in range(n):
            a = self.coeffs[i] if i < len(self.coeffs) else Fraction(0)
            b = other.coeffs[i] if i < len(other.coeffs) else Fraction(0)
            out.append(a - b)
        return Polynomial(out)

    def __neg__(self) -> "Polynomial":
        return Polynomial([-c for c in self.coeffs])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self.coeffs == other.coeffs

    def __hash__(self) -> int:
        return hash(tuple(self.coeffs))

    def __repr__(self) -> str:
        return f"Polynomial({self.coeffs!r})"

    def monic_part(self) -> "Polynomial":
        """Return this polynomial divided by its leading coefficient.

        For the zero polynomial this returns the zero polynomial.
        """
        if self.is_zero():
            return Polynomial([0])
        lead = self.coeffs[-1]
        return Polynomial([c / lead for c in self.coeffs])

    def derivative(self) -> "Polynomial":
        """First derivative.  The derivative of a constant (including zero) is ``0``."""
        if self.degree <= 0:
            return Polynomial([0])
        out = [self.coeffs[i] * i for i in range(1, len(self.coeffs))]
        return Polynomial(out)

    def floordiv(self, other: "Polynomial") -> "Polynomial":
        """Polynomial long division (quotient only).

        Assumes ``other`` is non-zero.  Raises :class:`ZeroDivisionError`
        otherwise.
        """
        if other.is_zero():
            raise ZeroDivisionError("polynomial division by zero")
        if self.degree < other.degree:
            return Polynomial([0])
        rem = list(self.coeffs)
        d = other.degree
        lead = other.coeffs[d]
        q = [Fraction(0)] * (self.degree - d + 1)
        for i in range(self.degree - d, -1, -1):
            if len(rem) - 1 < i + d:
                continue
            coef = rem[i + d] / lead
            q[i] = coef
            for j in range(d + 1):
                rem[i + j] -= coef * other.coeffs[j]
        return Polynomial(q)

    def mod(self, other: "Polynomial") -> "Polynomial":
        """Polynomial remainder (``self - other * (self // other)``)."""
        if other.is_zero():
            raise ZeroDivisionError("polynomial modulo by zero")
        if self.degree < other.degree:
            return Polynomial(self.coeffs)
        q = self.floordiv(other)
        return self - other * q


def sturm_sequence(poly: Polynomial) -> List[Polynomial]:
    """Return the canonical Sturm sequence of ``poly``.

    The sequence is ``[p0, p1, p2, ...]`` where ``p0 = poly`` and
    ``p1 = poly.derivative()``.  Each subsequent ``p_{k+1}`` is the negation of
    the remainder of ``p_{k-1}`` divided by ``p_k``.  The sequence terminates
    when a remainder of zero is reached.

    The leading coefficient of each non-final term is used to make that term
    monic *before* the next division.  This keeps the numerators and
    denominators from growing without bound (a well-known practical problem
    with naive Sturm sequences over the rationals) while preserving the sign
    pattern that Sturm's theorem relies on.

    Raises :class:`ValueError` if ``poly`` is the zero polynomial or has degree
    zero — Sturm's theorem requires a non-constant polynomial.
    """
    if poly.is_zero():
        raise ValueError("Sturm sequence is undefined for the zero polynomial")
    if poly.degree == 0:
        raise ValueError("Sturm sequence requires a non-constant polynomial")

    seq = [poly, poly.derivative()]
    while not seq[-1].is_zero():
        # If the remainder is about to be zero, stop — the current last element
        # is the GCD of (poly, poly').  Including a trailing zero would break
        # sign-change counting.
        prev = seq[-2]
        cur = seq[-1]
        # Normalise the divisor to be monic so coefficients stay small.
        cur_monic = cur.monic_part()
        rem = prev.mod(cur_monic)
        if rem.is_zero():
            break
        seq.append(-rem)
    return seq


def _sign(x: Fraction) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _sign_changes(seq: Sequence[Polynomial], x: Fraction) -> int:
    """Count sign changes in ``[p(x) for p in seq]``, skipping zeros."""
    count = 0
    prev = 0
    for p in seq:
        s = _sign(p(x))
        if s == 0:
            continue
        if prev != 0 and s != prev:
            count += 1
        prev = s
    return count


def count_real_roots(
    poly: Polynomial,
    a: Fraction = None,
    b: Fraction = None,
) -> int:
    """Count distinct real roots of ``poly`` in the half-open interval ``[a, b)``.

    Uses Sturm's theorem: the number of distinct real roots in ``[a, b)`` is
    ``V(a) - V(b)``, where ``V`` is the number of sign changes in the Sturm
    sequence evaluated at the point.

    Parameters
    ----------
    poly:
        Non-constant polynomial with rational coefficients.
    a, b:
        Rational endpoints.  Both default to ``None``, which is interpreted as
        ``-infinity`` and ``+infinity`` respectively.  When supplied they are
        converted to :class:`~fractions.Fraction`.

    The interval is half-open on the right: a root exactly at ``b`` is not
    counted, while a root exactly at ``a`` is.  This matches the standard
    statement of Sturm's theorem and avoids double-counting when isolating
    roots by bisection.
    """
    seq = sturm_sequence(poly)

    if a is None:
        va = _sign_changes_at_minus_infinity(seq)
    else:
        a = Fraction(a)
        va = _sign_changes(seq, a)

    if b is None:
        vb = _sign_changes_at_plus_infinity(seq)
    else:
        b = Fraction(b)
        vb = _sign_changes(seq, b)

    n = va - vb
    # Sturm's theorem counts roots in [a, b] (closed interval).  For the
    # half-open interval [a, b), subtract a root that sits exactly at b.
    if b is not None and poly(b) == 0:
        n -= 1
    return n


def _sign_changes_at_minus_infinity(seq: Sequence[Polynomial]) -> int:
    """Sign changes of the Sturm sequence as ``x -> -inf``.

    For a polynomial, the sign at ``-inf`` is the sign of its leading term
    evaluated at a large negative number, which is ``sign(lead) * (-1)^degree``.
    """
    signs = []
    for p in seq:
        if p.is_zero():
            continue
        d = p.degree
        lead = p.coeffs[d]
        s = _sign(lead)
        if d % 2 == 1:
            s = -s
        signs.append(s)
    return _count_sign_changes_from_list(signs)


def _sign_changes_at_plus_infinity(seq: Sequence[Polynomial]) -> int:
    """Sign changes of the Sturm sequence as ``x -> +inf``."""
    signs = []
    for p in seq:
        if p.is_zero():
            continue
        signs.append(_sign(p.coeffs[p.degree]))
    return _count_sign_changes_from_list(signs)


def _count_sign_changes_from_list(signs: Sequence[int]) -> int:
    count = 0
    prev = 0
    for s in signs:
        if s == 0:
            continue
        if prev != 0 and s != prev:
            count += 1
        prev = s
    return count


def _cauchy_bound(poly: Polynomial) -> Fraction:
    """A Cauchy bound on the magnitude of all roots of ``poly``.

    Returns ``1 + max(|a_i / a_n|)`` for ``i < n``, where ``a_n`` is the
    leading coefficient.  All real roots lie in ``[-B, B]``.
    """
    if poly.is_zero():
        raise ValueError("cannot bound roots of the zero polynomial")
    n = poly.degree
    lead = poly.coeffs[n]
    m = max((abs(poly.coeffs[i] / lead) for i in range(n)), default=Fraction(0))
    return 1 + m


def isolate_real_roots(
    poly: Polynomial,
    a: Optional[Fraction] = None,
    b: Optional[Fraction] = None,
) -> List[Tuple[Fraction, Fraction]]:
    """Isolate the distinct real roots of ``poly``.

    Returns a list of open intervals ``(lo, hi)`` such that each interval
    contains exactly one distinct real root of ``poly`` and the intervals are
    disjoint and ordered left to right.

    If ``a`` or ``b`` is supplied, only roots in ``[a, b)`` are isolated.
    Otherwise a Cauchy bound is used to obtain a finite starting interval.

    The endpoints of every returned interval are exact rationals
    (:class:`~fractions.Fraction`).

    Note: roots with even multiplicity are *not* reported, because Sturm's
    theorem counts distinct roots.  A polynomial ``(x-1)^2`` has one distinct
    real root, and this function returns a single isolating interval for it.
    """
    seq = sturm_sequence(poly)

    if a is None or b is None:
        bound = _cauchy_bound(poly)
        if a is None:
            a = -bound
        if b is None:
            b = bound
    a = Fraction(a)
    b = Fraction(b)

    intervals: List[Tuple[Fraction, Fraction]] = []
    _bisect(seq, poly, a, b, intervals)
    return intervals


def _bisect(
    seq: Sequence[Polynomial],
    poly: Polynomial,
    lo: Fraction,
    hi: Fraction,
    out: List[Tuple[Fraction, Fraction]],
) -> None:
    """Recursively bisect ``[lo, hi]`` until each sub-interval has at most one root."""
    v_lo = _sign_changes(seq, lo)
    v_hi = _sign_changes(seq, hi)
    n = v_lo - v_hi
    if n == 0:
        return
    if n == 1:
        out.append((lo, hi))
        return
    # n > 1: split and recurse.  The midpoint is an exact rational.
    mid = (lo + hi) / 2
    # If the midpoint is exactly a root, nudge it slightly so that bisection
    # does not produce a degenerate interval with the root at an endpoint.
    if poly(mid) == 0:
        # Shift the midpoint by a small rational amount toward the wider side.
        # This keeps the root strictly inside one of the two sub-intervals.
        step = (hi - lo) / 4
        mid = mid + step
    _bisect(seq, poly, lo, mid, out)
    _bisect(seq, poly, mid, hi, out)
