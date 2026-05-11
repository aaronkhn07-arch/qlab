"""
Periodic Rydberg blockade (PXP, PBC):
  1. enumerate valid states,
  2. count them (and check against Lucas numbers),
  3. group them into translation orbits,
  4. build the k = 0 momentum-sector basis.

Bit convention matches scarState.py: position i of the chain is the
(L-1-i)-th bit of the integer, so bit_at(s, 0) is the MSB.
"""

import numpy as np


def bit_at(s, i, L):
    return (s >> (L - 1 - i)) & 1


def is_valid_pbc(s, L):
    """No two adjacent 1s on the ring (includes the (L-1, 0) wrap bond)."""
    for i in range(L - 1):
        if ((s >> (L - 1 - i)) & 1) and ((s >> (L - 2 - i)) & 1):
            return False
    return not ((s & 1) and ((s >> (L - 1)) & 1))


def generate_pbc_basis(L):
    """Task 1: every valid PBC blockade configuration, as an integer list."""
    return [s for s in range(1 << L) if is_valid_pbc(s, L)]


def lucas(n):
    """Lucas number L_n. Equals the number of independent sets on C_n."""
    if n == 0:
        return 2
    a, b = 2, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def translate(s, L):
    """Cyclic shift of the chain by one site."""
    return (s >> 1) | ((s & 1) << (L - 1))


def build_orbits(basis, L):
    """Task 3: partition the PBC basis into translation orbits.

    Returns a list of (rep, period, members) where:
      rep     = smallest integer in the orbit (canonical representative),
      period  = orbit size (a divisor of L),
      members = tuple of the orbit's states in shift order starting from rep.
    """
    state_set = set(basis)
    seen = set()
    orbits = []
    for s in basis:
        if s in seen:
            continue
        members = []
        t = s
        while t not in seen:
            members.append(t)
            seen.add(t)
            t = translate(t, L)
        rep = min(members)
        k = members.index(rep)
        members = tuple(members[k:] + members[:k])
        orbits.append((rep, len(members), members))
        assert state_set.issuperset(members), "orbit escapes basis"
    orbits.sort(key=lambda o: o[0])
    return orbits


def build_k0_basis(orbits):
    """Task 4: one k=0 eigenstate per orbit, |k=0;r> = (1/sqrt(N)) Σ T^j |r>."""
    k0 = []
    for rep, period, members in orbits:
        amp = 1.0 / np.sqrt(period)
        coeffs = {m: amp for m in members}
        k0.append((rep, period, coeffs))
    return k0


def k0_basis_as_matrix(k0, basis):
    """Dense (D x D_k0) embedding of the k=0 basis into the full PBC basis."""
    idx = {s: i for i, s in enumerate(basis)}
    M = np.zeros((len(basis), len(k0)), dtype=np.float64)
    for j, (_, _, coeffs) in enumerate(k0):
        for s, c in coeffs.items():
            M[idx[s], j] = c
    return M


def _report(L):
    basis = generate_pbc_basis(L)
    orbits = build_orbits(basis, L)
    k0 = build_k0_basis(orbits)
    assert len(basis) == lucas(L), f"count mismatch at L={L}"
    assert sum(p for _, p, _ in orbits) == len(basis)
    M = k0_basis_as_matrix(k0, basis)
    assert np.allclose(M.T @ M, np.eye(len(k0))), "k=0 basis not orthonormal"
    return len(basis), len(orbits), len(k0)


if __name__ == "__main__":
    print(f"{'L':>3} {'|basis_PBC|':>12} {'Lucas(L)':>10} "
          f"{'#orbits':>9} {'dim(k=0)':>10}")
    for L in range(2, 17):
        n, n_orb, n_k0 = _report(L)
        print(f"{L:>3} {n:>12d} {lucas(L):>10d} {n_orb:>9d} {n_k0:>10d}")

    L = 6
    basis = generate_pbc_basis(L)
    orbits = build_orbits(basis, L)
    print(f"\nL={L} orbits (rep in binary, period, members):")
    for rep, period, members in orbits:
        members_bin = [format(m, f"0{L}b") for m in members]
        print(f"  {format(rep, f'0{L}b')}  period={period}  {members_bin}")
