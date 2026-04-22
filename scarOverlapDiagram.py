import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def bit_at(s, i, L):
    return (s >> (L - 1 - i)) & 1


def flip_bit(s, i, L):
    return s ^ (1 << (L - 1 - i))


def z2_index(L):
    s = 0
    for i in range(L):
        b = 1 if i % 2 == 0 else 0
        s = (s << 1) | b
    return s


def is_valid_pxp_state(s, L):
    # Open boundary conditions: only nearest-neighbor blockade constraints in the bulk.
    for i in range(L - 1):
        if bit_at(s, i, L) == 1 and bit_at(s, i + 1, L) == 1:
            return False
    return True


def generate_valid_basis(L):
    return [s for s in range(1 << L) if is_valid_pxp_state(s, L)]


def build_valid_basis_maps(L):
    basis = generate_valid_basis(L)
    state_to_idx = {s: i for i, s in enumerate(basis)}
    return basis, state_to_idx


def build_pxp_hamiltonian_constrained(L, basis, state_to_idx):
    rows = []
    cols = []
    data = []

    # In the constrained basis, flipping a bit is allowed iff the resulting state
    # is still in the blockade subspace.
    for col, s in enumerate(basis):
        for i in range(L):
            sp = flip_bit(s, i, L)
            if sp in state_to_idx:
                row = state_to_idx[sp]
                rows.append(row)
                cols.append(col)
                data.append(1.0)

    dim = len(basis)
    return csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64)


def z2_state_constrained(L, basis, state_to_idx):
    s = z2_index(L)
    if s not in state_to_idx:
        raise ValueError('|Z2> is not in the constrained basis.')
    psi = np.zeros(len(basis), dtype=np.float64)
    psi[state_to_idx[s]] = 1.0
    return psi


def scar_overlap_diagram(L=16, marker_size=26):
    basis, state_to_idx = build_valid_basis_maps(L)

    print(f"L = {L}")

    H = build_pxp_hamiltonian_constrained(L, basis, state_to_idx)
    psi_z2 = z2_state_constrained(L, basis, state_to_idx)

    # Full exact diagonalization. Practical for moderate L (e.g. L <= 16).
    Harray = H.toarray()
    energies, eigvecs = np.linalg.eigh(Harray)

    # Columns of eigvecs are eigenvectors |E_n>.
    overlaps = np.abs(eigvecs.T @ psi_z2) ** 2

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(energies, overlaps, s=marker_size)
    ax.set_xlabel('Energy $E_n$')
    ax.set_ylabel(r'$|\langle E_n | Z_2 \rangle|^2$')
    ax.set_title(f'Scar overlap diagram for open-chain PXP model (L={L})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return energies, overlaps, eigvecs, basis
scar_overlap_diagram(L=18, marker_size=8)

if __name__ == '__main__':
    # Start with open boundary conditions, as requested.
    scar_overlap_diagram(L=16, marker_size=28, annotate_top=10)
