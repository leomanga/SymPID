import multiprocessing as mp
from typing import Tuple
from numbers import Number
from tqdm import tqdm
import itertools
from sympy import (
    Poly,
    symbols,
    gcd,
    quo,
    nroots,
    sign,
    oo,
    I,
    re,
    Eq,
    expand,
    lambdify,
    And,
    Symbol,
    EXRAW,
    latex
)
from sympy.polys.specialpolys import random_poly
from sympy.abc import s
import mpmath
import timeit
import numpy as np


from plotting import (plot_hurwitz, plot_stability_region,
                      plot_hurwitz_subplots, plot_hurwitz_3d,
                      plot_hermite_biehler_3d)

# --- Core Algorithm Helpers ---


def get_even_odd_parts(poly: Poly) -> tuple[Poly, Poly]:  # type: ignore
    coeffs = poly.all_coeffs()
    n = poly.degree()

    start_even = n % 2

    start_odd = (n + 1) % 2

    pe_raw = coeffs[start_even::2]
    po_raw = coeffs[start_odd::2]

    def space_out(lst: list):
        if not lst:
            return []
        result = []
        for c in lst[:-1]:
            result.append(c)
            result.append(0)
        result.append(lst[-1])
        return result

    Pe = Poly(space_out(pe_raw), s) if pe_raw else Poly(0, s)  # type: ignore
    Po = Poly(space_out(po_raw), s) if po_raw else Poly(0, s)  # type: ignore

    return Pe, Po


def _evaluate_at_jw(poly: Poly, w: Symbol) -> Poly:
    return Poly(poly.subs(s, w * I), w)


def _build_pq_polynomials(
    N_poly: Poly, D_poly: Poly, kp_val: Poly, w: Symbol
) -> tuple[Poly, Poly, Poly, Poly]:
    De, Do = get_even_odd_parts(D_poly)
    Ne, No = get_even_odd_parts(N_poly)

    G = gcd(Ne, No)
    Ne_ = Poly(quo(Ne, G), s)
    No_ = Poly(quo(No, G), s)
    N_ = Poly(Ne_ + s * No_, s)

    Ne_jw = _evaluate_at_jw(Ne_, w)
    No_jw = _evaluate_at_jw(No_, w)
    De_jw = _evaluate_at_jw(De, w)
    Do_jw = _evaluate_at_jw(Do, w)
    Ne_jw_full = _evaluate_at_jw(Ne, w)
    No_jw_full = _evaluate_at_jw(No, w)

    p1 = -(w**2) * (Ne_jw * Do_jw - De_jw * No_jw)
    p2 = Ne_jw_full * Ne_jw + w**2 * No_jw_full * No_jw

    q1 = w * (De_jw * Ne_jw + w**2 * Do_jw * No_jw)
    q2 = w * (Ne_jw_full * Ne_jw + w**2 * No_jw_full * No_jw)

    q = q1 + kp_val * q2

    return p1, p2, q, N_


def _find_non_negative_roots(q_poly: Poly) -> tuple[int, list]:
    """
    Finds the non-negative roots of the polynomial q(w).
    Returns the number of such roots (l) and the list of roots including infinity.
    """
    roots_list = nroots(q_poly, n=8, maxsteps=100)
    tolerance = 1e-5
    
    # 1. Filter and extract real roots (with tolerance for 0) and SORT
    real_roots = sorted([float(re(r)) for r in roots_list if r.is_real and float(re(r)) >= -tolerance])
    
    odd_mult_roots = []
    if not real_roots:
        return 0, [oo]
    current_root = real_roots[0]
    count = 1
    
    # 2. Iterate through the sorted array
    for i in range(1, len(real_roots)):
        if abs(real_roots[i] - current_root) < tolerance:
            count += 1  # Same root (within tolerance)
        else:
            # 3. As soon as a different root is found, evaluate the previous group
            if count % 2 != 0:
                odd_mult_roots.append(max(0.0, current_root))
            
            # Reset counters for the new group
            current_root = real_roots[i]
            count = 1
            
    # 4. Final check for the last group remaining at the end of the loop
    if count % 2 != 0:
        odd_mult_roots.append(max(0.0, current_root))
        
    l = len(odd_mult_roots)
    odd_mult_roots.append(oo)
    return l, odd_mult_roots


def _calculate_sigma(poly: Poly) -> int:
    roots = nroots(poly)
    sig = 0
    for r in roots:
        rr = re(r)
        if rr > 0:
            sig -= 1
        elif rr < 0:
            sig += 1
    return sig


def _calculate_degrees_and_sigma(
    N: Poly, D: Poly, N_: Poly, kp_val: Number,
    ki: Symbol, kd: Symbol
) -> tuple[int, int, int]:
    delta = N.gen * D + (ki + kd * N.gen**2) * N + kp_val * N.gen * N
    n: int = delta.degree()
    m_ = N_.degree()
    if m_ == -oo:
        raise Exception("Unexpected error, probably N_ is a constant")
    sigmaN_ = _calculate_sigma(N_)
    return n, m_, sigmaN_  # type: ignore


def _determine_predefined_signs(
    p1: Poly, p2: Poly, real_roots: list, l: int
) -> list:
    """Determines the predefined signs for the i_strings based on root evaluations."""
    i_predef: list[None | int] = [None] * (l + 1)
    tolerance = 1e-5
    for idx in range(l):
        wt = real_roots[idx]
        p2_val = p2.subs(p1.gen, wt)
        if abs(float(p2_val)) < tolerance:
            p1_val = p1.subs(p1.gen, wt)
            i_predef[idx] = int(sign(p1_val))

    if p2.degree() + 2 < p1.degree():
        i_predef[l] = int(sign(p1.LC()))
    return i_predef


def _build_stability_inequalities(
    p1: Poly,
    p2: Poly,
    valid_strings: list,
    real_roots: list,
    ki: Symbol,
    kd: Symbol,
) -> list:
    """Builds the final stability inequalities based on valid strings."""
    ineq_poly_w = p1 + (ki - kd * p1.gen**2) * p2
    tolerance = 1e-5

    union = []
    for string in valid_strings:
        conds = []
        for idx, sym in enumerate(string):
            i_val = string[sym]
            if real_roots[idx] != oo:
                # Skip adding the inequality if p2_val is essentially zero (Step 3 of the paper)
                p2_val = p2.subs(p1.gen, real_roots[idx])
                if abs(float(p2_val)) >= tolerance:
                    cond = ineq_poly_w.subs(p1.gen, real_roots[idx]) * i_val
                    conds.append(cond)
            else:
                cond = ineq_poly_w.LC() * i_val
                conds.append(cond)
        union.append(conds)

    return union


# --- String Generators ---


def _find_valid_i_strings_backtracking(deg_sum, lp, l, target, i_predef):
    max_idx = l + 1 if deg_sum % 2 == 0 else l

    if max_idx == 0:
        return None

    coeffs = [0] * max_idx
    if max_idx > 0:
        coeffs[0] += lp
    for j in range(1, l):
        coeffs[j] += (-1) ** j * 2 * lp
    if deg_sum % 2 == 0:
        coeffs[l] += (-1) ** l * lp

    possibilities = []
    for idx in range(max_idx):
        if idx < len(i_predef) and i_predef[idx] is not None:
            possibilities.append([i_predef[idx]])
        elif idx == 0:
            possibilities.append([-1, 0, 1])
        else:
            possibilities.append([-1, 1])

    n_vars = len(coeffs)
    suffix_max = [0] * (n_vars + 1)
    suffix_min = [0] * (n_vars + 1)

    for i in range(n_vars - 1, -1, -1):
        c = coeffs[i]
        possible_values = [c * p for p in possibilities[i]]
        suffix_max[i] = max(possible_values) + suffix_max[i + 1]
        suffix_min[i] = min(possible_values) + suffix_min[i + 1]

    syms = symbols(f"i_0:{max_idx}")
    valid_strings = []

    def backtrack(idx, current_sum, current_combo):
        if (
            current_sum + suffix_max[idx] < target
            or current_sum + suffix_min[idx] > target
        ):
            # Pruned branch
            return

        if idx == n_vars:
            if current_sum == target:
                valid_strings.append(dict(zip(syms, current_combo)))
            return

        for val in possibilities[idx]:
            backtrack(
                idx + 1,
                current_sum + (coeffs[idx] * val),
                current_combo + [val],
            )

    backtrack(0, 0, [])

    return valid_strings


def _find_valid_i_strings_itertools(deg_sum, lp, l, target, i_predef):
    max_idx = l + 1 if deg_sum % 2 == 0 else l

    if max_idx == 0:
        return None

    possibilities = []
    for idx in range(max_idx):
        if idx < len(i_predef) and i_predef[idx] is not None:
            possibilities.append([i_predef[idx]])
        elif idx == 0:
            possibilities.append([-1, 0, 1])
        else:
            possibilities.append([-1, 1])

    syms = symbols(f"i_0:{max_idx}")
    eq = syms[0]
    for j in range(1, l):
        eq += (-1) ** j * 2 * syms[j]
    if deg_sum % 2 == 0:
        eq += (-1) ** l * syms[l]
    eq *= lp
    sym_eq = Eq(eq, target)

    valid_strings = []
    for combo in itertools.product(*possibilities):
        subs_map = dict(zip(syms, combo))
        if sym_eq.subs(subs_map):
            valid_strings.append(subs_map)

    return valid_strings


# --- Main Functions ---


def calculate_pid_stability_conditions_backtracking(N_poly, D_poly, kp_val):
    """
    Enhanced version of PID stability conditions calculator using backtracking.
    """
    ki, kd = symbols("k_i k_d")
    w = symbols("omega")

    p1, p2, q_poly, N_ = _build_pq_polynomials(N_poly, D_poly, kp_val, w)
    l, real_roots = _find_non_negative_roots(q_poly)
    n, m_, sigmaN_ = _calculate_degrees_and_sigma(
        N_poly, D_poly, N_, kp_val, ki, kd
    )
    last_prod = (-1) ** (l - 1) * sign(q_poly.LC())
    i_predef = _determine_predefined_signs(p1, p2, real_roots, l)

    deg_sum = m_ + n
    valid_strings = _find_valid_i_strings_backtracking(
        deg_sum, last_prod, l, n - sigmaN_, i_predef
    )

    if not valid_strings:
        return None

    return _build_stability_inequalities(
        p1, p2, valid_strings, real_roots, ki, kd
    )


def calculate_pid_stability_conditions_itertools(N_poly, D_poly, kp_val):
    """
    Simple version of PID stability conditions calculator using itertools.
    """
    ki, kd = symbols("k_i k_d")
    w = symbols("omega")

    p1, p2, q_poly, N_ = _build_pq_polynomials(N_poly, D_poly, kp_val, w)
    l, real_roots = _find_non_negative_roots(q_poly)
    n, m_, sigmaN_ = _calculate_degrees_and_sigma(
        N_poly, D_poly, N_, kp_val, ki, kd
    )
    last_prod = (-1) ** (l - 1) * sign(q_poly.LC())
    i_predef = _determine_predefined_signs(p1, p2, real_roots, l)

    deg_sum = m_ + n
    valid_strings = _find_valid_i_strings_itertools(
        deg_sum, last_prod, l, n - sigmaN_, i_predef
    )

    if not valid_strings:
        return None

    return _build_stability_inequalities(
        p1, p2, valid_strings, real_roots, ki, kd
    )


def build_closed_loop_characteristic_poly(
    N: Poly, D: Poly, kp=None
) -> Tuple[Poly, Symbol, Symbol, Symbol]:
    kd_sym, kp_sym, ki_sym = symbols("kd kp ki")
    if kp is not None:
        kp_sym = kp
    C_N = kd_sym * s**2 + kp_sym * s + ki_sym
    C_D = Poly(s, s)
    res = C_D * D + N * C_N

    return (res, kp_sym, ki_sym, kd_sym)

