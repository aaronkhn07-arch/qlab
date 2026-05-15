"""
Analytical k=0 and reflection-symmetric k=0 sector dimensions for the periodic PXP chain.

Brute-force enumeration via is infeasible once L is large
(2^L checks). Instead, we count translation orbits directly
with Burnside's lemma:

    N_{k=0} = (1/L) * sum_{d | L} phi(L/d) * Lucas(d)

Reasoning:
  - The translation group is cyclic of order L.
  - For shift T^k, a state is fixed, meaning unchanged by the periodic shifts, iff its period divides gcd(k, L).
  - Such states are determined by their first d := gcd(k, L) bits, which
    must form a valid PBC blockade configuration on a ring of length d.
  - The number of valid PBC blockade configurations on a ring of length d
    is the Lucas number Lucas(d) (which matches fib2 in periodicPXP.py).
  - Counting how many k in {0,...,L-1} give gcd(k, L) = d yields phi(L/d).

Reflection symmetry (D_0^+):
  - Reflection R sends position i in the chain to position L-1-i.
  - R does not commute with T but satisfies R T R = T^{-1}; together they
    generate the dihedral group D_L. R preserves the k=0 subspace because
    it sends a translation orbit to another translation orbit.
  - Within the k=0 sector R has eigenvalues +/- 1. The +1 eigenspace D_0^+ is
    spanned by:
        * |O>                          for each self-reflective orbit (R(O) = O),
        * (|O> + |R(O)>) / sqrt(2)     for each pair O <-> R(O) with R(O) != O.
  - Counting: dim D_0^+ = (dim D_0 + N_sr) / 2, where N_sr is the number of
    self-reflective orbits.
"""

import numpy as np


def lucas(n):
    """Lucas number L_n: L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}.

    Equals the number of valid PBC blockade configurations on a ring of
    length n (same value computed by fib2 in periodicPXP.py).
    """
    if n == 0:
        return 2
    a, b = 2, 1  # (L_0, L_1)
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def fib(n):
    """Standard Fibonacci: F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def euler_phi(n):
    """Euler totient phi(n): count of integers in [1, n] coprime to n.

    Standard trial-division implementation; fine since we only call it on
    divisors of L, which is small.
    """
    result = n
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            # strip out all copies of the prime p
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1
    if m > 1:
        # m is a prime factor larger than sqrt(n)
        result -= result // m
    return result


def divisors(n):
    """All positive divisors of n, in ascending order."""
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            if i != n // i:
                out.append(n // i)
        i += 1
    return sorted(out)


def is_prime(n):
    """Trial-division primality test (only used on L itself, so very cheap)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    p = 3
    while p * p <= n:
        if n % p == 0:
            return False
        p += 2
    return True


def k0_dimension(L):
    """Dimension of the k=0 momentum sector for the periodic PXP chain of length L.

    Uses Burnside's lemma over the cyclic translation group: the number
    of orbits equals the average number of states fixed by each element.
    For each divisor d of L, exactly phi(L/d) translations have a fixed
    set of size Lucas(d), giving the formula below.
    """
    total = 0
    for d in divisors(L):
        # phi(L/d) shifts have gcd(k, L) = d; each fixes Lucas(d) states.
        total += euler_phi(L // d) * lucas(d)
    # Burnside guarantees L divides the sum exactly.
    return total // L


# ----------------------------------------------------------------------
# Reflection symmetry: explicit construction of the D_0^+ basis.
# ----------------------------------------------------------------------


def reflect(s, L):
    """Reflection R applied to a bitstring on a ring of length L.

    Position i in the chain corresponds to bit (L-1-i) of the integer
    (MSB-first convention used throughout the project). Reflection swaps
    position i with position L-1-i, which is equivalent to reversing the
    L-bit binary representation of s.
    """
    r = 0
    for i in range(L):
        # bit i of s -> position (L-1-i) of the reflected integer
        if (s >> i) & 1:
            r |= 1 << (L - 1 - i)
    return r


def build_k0_plus_basis(orbits, L):
    """Construct the D_0^+ basis: k=0 vectors with reflection eigenvalue +1.

    For each translation orbit O, R either maps O to itself ("self-reflective")
    or to a distinct partner orbit O'. The +1 eigenstates are:

      - self-reflective:  |O>                    (already an R-eigenstate)
      - paired:           (|O> + |O'>)/sqrt(2)   (symmetric combination)

    Returns a list of (rep, coeffs) tuples, where coeffs is a dict
    {bitstring: amplitude} that embeds the D_0^+ vector into the full PBC
    basis. rep is taken as the smallest member across the orbit(s) involved,
    purely to give the output a deterministic ordering.
    """
    # Fast lookup from any bitstring to the index of the orbit containing it.
    state_to_orbit_idx = {}
    for i, (_, _, members) in enumerate(orbits):
        for s in members:
            state_to_orbit_idx[s] = i

    handled = set()  # orbit indices already absorbed into D_0^+
    plus = []

    for a, (rep_a, tau_a, members_a) in enumerate(orbits):
        if a in handled:
            continue
        # Reflect a single representative to identify the partner orbit;
        # because R commutes with the k=0 projector, this fully determines
        # which orbit R sends O_a to.
        b = state_to_orbit_idx[reflect(rep_a, L)]

        if b == a:
            # Self-reflective: |O_a> is already a +1 eigenvector of R.
            amp = 1.0 / np.sqrt(tau_a)
            coeffs = {s: amp for s in members_a}
            plus.append((rep_a, coeffs))
            handled.add(a)
        else:
            # Distinct partner: take the symmetric combination
            # (|O_a> + |O_b>)/sqrt(2). The antisymmetric combination would
            # belong to D_0^- and is discarded here.
            rep_b, tau_b, members_b = orbits[b]
            amp_a = 1.0 / np.sqrt(2.0 * tau_a)
            amp_b = 1.0 / np.sqrt(2.0 * tau_b)
            coeffs = {s: amp_a for s in members_a}
            coeffs.update({s: amp_b for s in members_b})
            plus.append((min(rep_a, rep_b), coeffs))
            handled.add(a)
            handled.add(b)

    plus.sort(key=lambda p: p[0])
    return plus


def count_self_reflective_orbits(orbits, L):
    """Number of translation orbits O with R(O) = O. Enumerative; for small L."""
    state_to_orbit_idx = {}
    for i, (_, _, members) in enumerate(orbits):
        for s in members:
            state_to_orbit_idx[s] = i
    return sum(
        1 for i, (rep, _, _) in enumerate(orbits)
        if state_to_orbit_idx[reflect(rep, L)] == i
    )


def k0_plus_dimension_odd_prime(L):
    """Analytical (dim D_0^+, N_sr) for odd prime L.

    For odd L, every reflection in D_L fixes exactly one site of the chain.
    A valid PBC state s fixed by such a reflection is determined by its
    first (L+1)/2 bits, with the additional constraint that bit (L-1)/2 is
    zero (it would otherwise be adjacent to its own reflected copy). The
    count of such states is F_{(L+3)/2}.

    For prime L the only non-primitive valid state is |0...0>. A primitive
    self-reflective orbit contributes exactly L fixed (state, reflection)
    pairs (each of the L reflections fixes one of its L members). The zero
    orbit contributes L by itself. Burnside over the reflections then gives

        L * F_{(L+3)/2}  =  (N_sr - 1) * L  +  L
        =>  N_sr  =  F_{(L+3)/2}.

    Combined with dim D_0^+ = (dim D_0 + N_sr) / 2, we get a closed form.
    """
    if not (L > 2 and L % 2 == 1 and is_prime(L)):
        raise ValueError("this analytical formula assumes odd prime L > 2")
    N_k0 = k0_dimension(L)
    N_sr = fib((L + 3) // 2)
    return (N_k0 + N_sr) // 2, N_sr


# ----------------------------------------------------------------------
# Overarching driver — single entry point called from periodicPXP.py main.
# ----------------------------------------------------------------------


def report_pxp_sector_dimensions(L, verify_with=None):
    """Print dim D_0 (k=0) and dim D_0^+ (k=0, reflection +1) for length L.

    Parameters
    ----------
    L : int
        Chain length. Any L works for dim D_0; the closed-form dim D_0^+ is
        currently only implemented for odd prime L (which covers L=101).
    verify_with : iterable of small L's, optional
        For each small L provided, also build the orbits and the D_0^+ basis
        explicitly via periodicPXP.build_orbits, and check that the analytical
        formulas agree with the enumerated counts. This is a sanity gate, not
        the main computation.
    """
    # --- Main computation: analytical sector sizes for the target L. -----
    print(f"PXP sector dimensions for L = {L}")
    print("-" * 60)

    # k = 0 sector via Burnside on translations.
    total = 0
    for d in divisors(L):
        phi = euler_phi(L // d)
        Ld = lucas(d)
        total += phi * Ld
        print(f"  d = {d:>4}: phi(L/d) = {phi:>4}, Lucas(d) = {Ld}")
    N_k0 = total // L

    print("-" * 60)
    print(f"  Lucas(L) (total valid PBC states) = {lucas(L)}")
    print(f"  dim D_0   (k=0)                   = {N_k0}")

    # Reflection-symmetric subsector D_0^+ via Burnside on the dihedral group.
    if L > 2 and L % 2 == 1 and is_prime(L):
        N_plus, N_sr = k0_plus_dimension_odd_prime(L)
        print(f"  # self-reflective orbits          = {N_sr}  (= F_{(L+3)//2})")
        print(f"  dim D_0^+ (k=0, R = +1)           = {N_plus}")
    else:
        N_plus = None
        print("  (closed-form dim D_0^+ only implemented for odd prime L)")

    # --- Optional sanity check against enumeration for small L. ----------
    if verify_with:
        from periodicPXP import generate_pbc_basis, build_orbits
        print("-" * 60)
        print("verification against enumeration:")
        for L_small in verify_with:
            basis = generate_pbc_basis(L_small)
            orbits = build_orbits(basis, L_small)
            enum_k0 = len(orbits)
            enum_plus = len(build_k0_plus_basis(orbits, L_small))
            enum_sr = count_self_reflective_orbits(orbits, L_small)
            # Use analytical formula only when applicable.
            ana_k0 = k0_dimension(L_small)
            if L_small > 2 and L_small % 2 == 1 and is_prime(L_small):
                ana_plus, ana_sr = k0_plus_dimension_odd_prime(L_small)
                tag = f"D_0={enum_k0}/{ana_k0}, D_0^+={enum_plus}/{ana_plus}, N_sr={enum_sr}/{ana_sr}"
            else:
                # Formula-free analytical D_0^+ not implemented; just show enumeration.
                tag = f"D_0={enum_k0}/{ana_k0}, D_0^+={enum_plus} (enum only), N_sr={enum_sr}"
            print(f"  L = {L_small:>3}: {tag}")

    return N_k0, N_plus


if __name__ == "__main__":
    # Verify on a few small odd primes, then report L=101.
    report_pxp_sector_dimensions(101, verify_with=[5, 7, 11, 13])