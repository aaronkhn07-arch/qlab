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
    """No two adjacent 1s on the ring (includes the (L-1, 0))."""
    for i in range(L - 1):
        if ((s >> (L - 1 - i)) & 1) and ((s >> (L - 2 - i)) & 1):
            return False
    return not ((s & 1) and ((s >> (L - 1)) & 1))


def generate_pbc_basis(L):
    """Task 1: every valid PBC blockade configuration, as an integer list."""
    """We might turn this into DNC later on for efficiency"""
    return [s for s in range(1 << L) if is_valid_pbc(s, L)]


def fib2(n):
    """The fib formula we found (F_{n-1} + F_{n+1})"""
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
      rep = smallest integer in the equivalence class
      period = orbit size (shown to be period of each element),
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
        members = tuple(members[k:] + members[:k]) #order so that min rep is first. easier to check that we've done this correctly
        orbits.append((rep, len(members), members)) 
        assert state_set.issuperset(members), "orbit escapes basis" # sanity check
    orbits.sort(key=lambda o: o[0]) #sorts by min val
    return orbits

def build_k0_basis(orbits):
    """Task 4: one k=0 eigenstate per orbit, |k=0;r> = (1/sqrt(N)) sum T^j |r>."""
    k0 = []
    for rep, period, members in orbits:
        amp = 1.0 / np.sqrt(period) #also why we need period
        coeffs = {m: amp for m in members} #js package everything as member: coefficient because for the matrix we js need M[i, j] = coef
        k0.append((rep, period, coeffs)) 
    return k0


def k0_basis_as_matrix(k0, basis):
    """Embedding of the k=0 basis into the full PBC basis. Creates matrix M whose columns are the k = 0 basis vectors"""
    idx = {s: i for i, s in enumerate(basis)}
    M = np.zeros((len(basis), len(k0)), dtype=np.float64)
    for j, (_, _, coeffs) in enumerate(k0):
        for s, c in coeffs.items():
            M[idx[s], j] = c
    return M

def flip_bit(s, i, L):
    """Flip site i, using the same MSB convention as bit_at."""
    return s ^ (1 << (L - 1 - i))

def z2_index(L):
    """Return integer corresponding to |Z2> = |1010...>."""
    s = 0
    for i in range(L):
        b = 1 if i % 2 == 0 else 0
        s = (s << 1) | b
    return s

def build_pxp_hamiltonian_k0(orbits, L):
    """
    Task 5: Construct the PXP Hamiltonian in the reduced k=0 basis.

    The k=0 basis vectors are orbit-superpositions:
        |O_a> = (1/sqrt(tau_a)) sum_{s in O_a} |s>

    Matrix element:
        H_{ba} = <O_b|H|O_a> = 1/sqrt(tau_a * tau_b) sum_{u in O_b} sum _{s in O_a} <u|H|s>
        and of course <u|H|s> = 1 iff u can be obtained from s via one allowed bit flip since H = PXP
        and 0 otherwise. 

    We compute this by:
      - taking every state s in orbit a,
      - flipping each site i,
      - checking whether the flipped state is valid and belongs to some orbit b,
      - adding the normalization factor 1/sqrt(tau_a tau_b).
    """

    dim = len(orbits)
    H0 = np.zeros((dim, dim), dtype=np.float64)

    # Map every bitstring state to its orbit index.
    state_to_orbit_idx = {}
    for orbit_idx, (_, _, members) in enumerate(orbits):
        for s in members:
            state_to_orbit_idx[s] = orbit_idx

    for a_idx, (_, tau_a, members_a) in enumerate(orbits):
        for s in members_a:
            for i in range(L):
                sp = flip_bit(s, i, L)

                # If sp is not in the valid PBC basis, then the flip is forbidden.
                if sp not in state_to_orbit_idx:
                    continue

                b_idx = state_to_orbit_idx[sp]
                tau_b = orbits[b_idx][1]

                H0[b_idx, a_idx] += 1.0 / np.sqrt(tau_a * tau_b)

    # # Sanity check: Hamiltonian should be Hermitian/symmetric.
    # assert np.allclose(H0, H0.T), "H0 is not symmetric; something is wrong."

    return H0


def z2_state_k0(orbits, L):
    """
    Construct the k=0 version of |Z2>.

    In the reduced k=0 basis, |Z2> is represented by the orbit containing
    the product state |1010...>. Since each orbit basis vector is already
    normalized, this is just a one-hot vector.
    """

    z2 = z2_index(L)

    for idx, (_, _, members) in enumerate(orbits):
        if z2 in members:
            psi = np.zeros(len(orbits), dtype=np.float64)
            psi[idx] = 1.0
            return psi


def diagonalize_k0_and_z2_overlaps(orbits, L, plot=False):
    """
    Task 6: Diagonalize H0 and compute overlaps with the k=0 |Z2> state.

    Returns:
        evals: eigenvalues of H0
        overlaps: |<E_n | Z2>|^2
        evecs: eigenvectors of H0
        H0: reduced Hamiltonian
    """

    H0 = build_pxp_hamiltonian_k0(orbits, L)

    evals, evecs = np.linalg.eigh(H0)

    psi_z2 = z2_state_k0(orbits, L)

    overlaps = np.abs(evecs.T @ psi_z2) ** 2

    if plot:
        import matplotlib.pyplot as plt
        overlaps_plot = overlaps[overlaps > 1e-14]
        evals_plot = evals[overlaps > 1e-14]

        plt.figure(figsize=(8, 5))
        plt.scatter(evals_plot, overlaps_plot, s=25)
        plt.yscale("log")
        plt.xlabel("Energy")
        plt.ylabel(r"$|\langle E_n | Z_2 \rangle|^2$")
        plt.title(f"PXP scar overlap plot in k=0 sector, L={L}")
        plt.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.show()
        
        print("sum overlaps =", np.sum(overlaps))
        print("max overlap =", np.max(overlaps))
        print("number nonzero-ish =", np.sum(overlaps > 1e-14))

    return evals, overlaps, evecs, H0

if __name__ == "__main__":
    L = 26

    basis = generate_pbc_basis(L)
    orbits = build_orbits(basis, L)

    diagonalize_k0_and_z2_overlaps(orbits, L, plot=True)