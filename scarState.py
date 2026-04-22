import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import expm_multiply
import time
import cProfile
import pstats
import io
from pstats import SortKey

def bit_at(s, i, L):
    return (s >> (L - 1 - i)) & 1


def flip_bit(s, i, L):
    return s ^ (1 << (L - 1 - i))


def z_value(bit):
    return 1 if bit == 0 else -1


def z2_index(L):

    s = 0
    for i in range(L):
        b = 1 if i % 2 == 0 else 0
        s = (s << 1) | b
    return s


def entropy_half_chain(psi, L):

    nA = L // 2
    dimA = 1 << nA
    dimB = 1 << (L - nA)

    M = psi.reshape(dimA, dimB)
    svals = np.linalg.svd(M, compute_uv=False)
    p = svals**2
    p = p[p > 1e-14]
    return -np.sum(p * np.log2(p))



def is_valid_pxp_state(s, L):
    for i in range(L - 1):
        if bit_at(s, i, L) == 1 and bit_at(s, i + 1, L) == 1:
            return False
    return True


def generate_valid_basis(L):
    return [s for s in range(1 << L) if is_valid_pxp_state(s, L)]


def fibonacci_subspace_dimension(L):
    a, b = 0, 1
    for _ in range(L + 2):
        a, b = b, a + b
    return a


def build_valid_basis_maps(L):
    basis = generate_valid_basis(L)
    state_to_idx = {s: i for i, s in enumerate(basis)}
    return basis, state_to_idx


def z2_state_constrained(L, basis, state_to_idx):
    s = z2_index(L)
    if s not in state_to_idx:
        raise ValueError("|Z2> is not in the constrained basis.")
    psi = np.zeros(len(basis), dtype=np.complex128)
    psi[state_to_idx[s]] = 1.0
    return psi



def build_pxp_hamiltonian_constrained(L, basis, state_to_idx):

    rows = []
    cols = []
    data = []

    for col, s in enumerate(basis):
        for i in range(L):
            sp = flip_bit(s, i, L)
            if sp in state_to_idx:
                row = state_to_idx[sp]
                rows.append(row)
                cols.append(col)
                data.append(1.0)

    dim = len(basis)
    H = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)
    return H


def build_mixed_field_ising_hamiltonian(L, J=1.0, hx=1.0, hz=0.5):

    dim = 1 << L

    diag = np.zeros(dim, dtype=np.float64)
    for s in range(dim):
        z = np.array([z_value(bit_at(s, i, L)) for i in range(L)], dtype=np.int8)
        diag[s] = J * np.sum(z[:-1] * z[1:]) + hz * np.sum(z)

    H = diags(diag, format="csr", dtype=np.complex128)

    rows = []
    cols = []
    data = []
    for s in range(dim):
        for i in range(L):
            sp = flip_bit(s, i, L)
            rows.append(sp)
            cols.append(s)
            data.append(hx)

    Hx = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)
    return H + Hx


def domain_wall_diagonal_full(L, density=True):

    dim = 1 << L
    vals = np.zeros(dim, dtype=np.float64)

    for s in range(dim):
        total = 0.0
        for i in range(L - 1):
            zi = z_value(bit_at(s, i, L))
            zip1 = z_value(bit_at(s, i + 1, L))
            total += 0.5 * (1.0 - zi * zip1)
        if density:
            total /= (L - 1)
        vals[s] = total

    return vals


def domain_wall_diagonal_constrained(L, basis, density=True):

    vals = np.zeros(len(basis), dtype=np.float64)

    for k, s in enumerate(basis):
        total = 0.0
        for i in range(L - 1):
            zi = z_value(bit_at(s, i, L))
            zip1 = z_value(bit_at(s, i + 1, L))
            total += 0.5 * (1.0 - zi * zip1)
        if density:
            total /= (L - 1)
        vals[k] = total

    return vals


def embed_constrained_state(psi_constrained, basis, L):
    psi_full = np.zeros(1 << L, dtype=np.complex128)
    for amp, s in zip(psi_constrained, basis):
        psi_full[s] = amp
    return psi_full



def evolve_and_measure_full(H, psi0, times, L, density=True):
    dw_diag = domain_wall_diagonal_full(L, density=density)

    states = expm_multiply(
        -1j * H,
        psi0,
        start=times[0],
        stop=times[-1],
        num=len(times),
        endpoint=True
    )

    dw = np.real(np.einsum("td,d,td->t", states.conj(), dw_diag, states))
    ent = np.array([entropy_half_chain(states[k], L) for k in range(len(times))])

    return dw, ent


def evolve_and_measure_constrained(H, psi0, times, L, basis, density=True):
    dw_diag = domain_wall_diagonal_constrained(L, basis, density=density)

    states = expm_multiply(
        -1j * H,
        psi0,
        start=times[0],
        stop=times[-1],
        num=len(times),
        endpoint=True
    )

    dw = np.real(np.einsum("td,d,td->t", states.conj(), dw_diag, states))

    ent = []
    for k in range(len(times)):
        psi_full = embed_constrained_state(states[k], basis, L)
        ent.append(entropy_half_chain(psi_full, L))
    ent = np.array(ent)

    return dw, ent



def run_quench_experiment(
    L=16,
    tmax=30.0,
    nt=601,
    J=1.0,
    hx=1.0,
    hz=0.5,
    density=True
):
    times = np.linspace(0.0, tmax, nt)
    begintime = time.time()
    # Constrained PXP objects
    basis, state_to_idx = build_valid_basis_maps(L)
    print(f"L = {L}")
    print(f"Constrained PXP dimension = {len(basis)}")
    print(f"Expected Fibonacci count  = {fibonacci_subspace_dimension(L)}")

    psi0_pxp = z2_state_constrained(L, basis, state_to_idx)
    H_pxp = build_pxp_hamiltonian_constrained(L, basis, state_to_idx)

    zeroCnt = 0
    zeroCnt = H_pxp.nnz
    print (f"{zeroCnt/len(basis)**2:.3f}")
    
    print(f"Number of zero rows in H_pxp: {zeroCnt}")

    # # Full-space Ising objects
    # psi0_ising = np.zeros(1 << L, dtype=np.complex128)
    # psi0_ising[z2_index(L)] = 1.0
    # H_ising = build_mixed_field_ising_hamiltonian(L, J=J, hx=hx, hz=hz)

    dw_pxp, ent_pxp = evolve_and_measure_constrained(
        H_pxp, psi0_pxp, times, L, basis, density=density
    )
    # dw_ising, ent_ising = evolve_and_measure_full(
    #     H_ising, psi0_ising, times, L, density=density
    # )
    endtime = time.time() - begintime
    print (f"Time: {endtime:.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ylabel_dw = "Domain wall density" if density else "Domain wall count"

    axes[0].plot(times, dw_pxp, label="PXP (constrained basis)", linewidth=2)
    #axes[0].plot(times, dw_ising, label=f"Mixed-field Ising\n(J={J}, hx={hx}, hz={hz})", linewidth=2)
    axes[0].set_ylabel(ylabel_dw)
    axes[0].set_title(f"Quantum quench from |Z2> for L={L}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(times, ent_pxp, label="PXP (constrained basis)", linewidth=2)
    #axes[1].plot(times, ent_ising, label=f"Mixed-field Ising\n(J={J}, hx={hx}, hz={hz})", linewidth=2)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Half-chain entanglement entropy (bits)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return times, dw_pxp, ent_pxp

if __name__ == "__main__":
    run_quench_experiment(
        L=20,
        tmax=30.0,
        nt=100,
        J=1.0,
        hx=1.0,
        hz=0.5,
        density=True
    )
