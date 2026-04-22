import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
import time


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


def entropy_half_chain(psi_full, L):
    nA = L // 2
    dimA = 1 << nA
    dimB = 1 << (L - nA)

    M = psi_full.reshape(dimA, dimB)
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


def build_valid_basis_maps(L):
    basis = generate_valid_basis(L)
    state_to_idx = {s: i for i, s in enumerate(basis)}
    return basis, state_to_idx


def z2_state_constrained(L, basis, state_to_idx):
    s = z2_index(L)
    psi = np.zeros(len(basis), dtype=np.complex128)
    psi[state_to_idx[s]] = 1.0
    return psi


def build_pxp_hamiltonian_constrained(L, basis, state_to_idx):
    rows, cols, data = [], [], []

    for col, s in enumerate(basis):
        for i in range(L):
            sp = flip_bit(s, i, L)
            if sp in state_to_idx:
                rows.append(state_to_idx[sp])
                cols.append(col)
                data.append(1.0)

    dim = len(basis)
    return csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)


def domain_wall_diagonal_constrained(L, basis):
    vals = np.zeros(len(basis), dtype=np.float64)
    for k, s in enumerate(basis):
        total = 0.0
        for i in range(L - 1):
            zi = z_value(bit_at(s, i, L))
            zip1 = z_value(bit_at(s, i + 1, L))
            total += 0.5 * (1.0 - zi * zip1)
        vals[k] = total / (L - 1)
    return vals


def embed_constrained_state(psi_constrained, basis, L):
    psi_full = np.zeros(1 << L, dtype=np.complex128)
    for amp, s in zip(psi_constrained, basis):
        psi_full[s] = amp
    return psi_full


def fidelity_expm(H, psi0, times):
    states = expm_multiply(
        -1j * H,
        psi0,
        start=times[0],
        stop=times[-1],
        num=len(times),
        endpoint=True
    )
    return np.abs(states @ psi0.conj())**2


def domain_wall_expm(H, psi0, times, L, basis):
    dw_diag = domain_wall_diagonal_constrained(L, basis)
    states = expm_multiply(
        -1j * H,
        psi0,
        start=times[0],
        stop=times[-1],
        num=len(times),
        endpoint=True
    )
    return np.real(np.einsum("td,d,td->t", states.conj(), dw_diag, states))


def entanglement_expm(H, psi0, times, L, basis):
    states = expm_multiply(
        -1j * H,
        psi0,
        start=times[0],
        stop=times[-1],
        num=len(times),
        endpoint=True
    )
    return np.array([
        entropy_half_chain(embed_constrained_state(states[k], basis, L), L)
        for k in range(len(times))
    ])

def diagonalize_dense(H):
    evals, evecs = np.linalg.eigh(H.toarray())
    return evals, evecs


def evolve_diag(evals, evecs, psi0, times):
    coeffs = evecs.conj().T @ psi0
    phases = np.exp(-1j * np.outer(times, evals))
    return (phases * coeffs[None, :]) @ evecs.conj().T


def fidelity_diag(H, psi0, times):
    evals, evecs = diagonalize_dense(H)
    states = evolve_diag(evals, evecs, psi0, times)
    return np.abs(states @ psi0.conj())**2


def domain_wall_diag(H, psi0, times, L, basis):
    dw_diag = domain_wall_diagonal_constrained(L, basis)
    evals, evecs = diagonalize_dense(H)
    states = evolve_diag(evals, evecs, psi0, times)
    return np.real(np.einsum("td,d,td->t", states.conj(), dw_diag, states))


def entanglement_diag(H, psi0, times, L, basis):
    evals, evecs = diagonalize_dense(H)
    states = evolve_diag(evals, evecs, psi0, times)
    return np.array([
        entropy_half_chain(embed_constrained_state(states[k], basis, L), L)
        for k in range(len(times))
    ])


def time_call(func, *args):
    t0 = time.perf_counter()
    out = func(*args)
    return time.perf_counter() - t0, out

def plot_observables_vs_time(L=12, tmax=30, nt=200, method="expm"):
    times = np.linspace(0, tmax, nt)

    basis, state_to_idx = build_valid_basis_maps(L)
    psi0 = z2_state_constrained(L, basis, state_to_idx)
    H = build_pxp_hamiltonian_constrained(L, basis, state_to_idx)

    if method == "expm":
        fidelity = fidelity_expm(H, psi0, times)
        domain_wall = domain_wall_expm(H, psi0, times, L, basis)
        entanglement = entanglement_expm(H, psi0, times, L, basis)
        method_label = "expm_multiply"
    elif method == "diag":
        fidelity = fidelity_diag(H, psi0, times)
        domain_wall = domain_wall_diag(H, psi0, times, L, basis)
        entanglement = entanglement_diag(H, psi0, times, L, basis)
        method_label = "direct diagonalization"

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(times, fidelity, linewidth=2)
    axes[0].set_ylabel("Fidelity")
    axes[0].set_title(f"PXP observables vs time for L={L} ({method_label})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, domain_wall, linewidth=2)
    axes[1].set_ylabel("Domain-wall density")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, entanglement, linewidth=2)
    axes[2].set_xlabel("Physical time t")
    axes[2].set_ylabel("Half-chain entanglement")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_benchmarks():
    L_vals = np.arange(8, 16)

    times_fid = np.linspace(0, 30, 40)
    times_dw = np.linspace(0, 30, 40)
    times_ent = np.linspace(0, 30, 16)

    expm_fid_times = []
    diag_fid_times = []

    expm_dw_times = []
    diag_dw_times = []

    expm_ent_times = []
    diag_ent_times = []

    for L in L_vals:
        basis, state_to_idx = build_valid_basis_maps(L)
        psi0 = z2_state_constrained(L, basis, state_to_idx)
        H = build_pxp_hamiltonian_constrained(L, basis, state_to_idx)

        t_expm_fid, _ = time_call(fidelity_expm, H, psi0, times_fid)
        t_diag_fid, _ = time_call(fidelity_diag, H, psi0, times_fid)

        t_expm_dw, _ = time_call(domain_wall_expm, H, psi0, times_dw, L, basis)
        t_diag_dw, _ = time_call(domain_wall_diag, H, psi0, times_dw, L, basis)

        t_expm_ent, _ = time_call(entanglement_expm, H, psi0, times_ent, L, basis)
        t_diag_ent, _ = time_call(entanglement_diag, H, psi0, times_ent, L, basis)

        expm_fid_times.append(t_expm_fid)
        diag_fid_times.append(t_diag_fid)

        expm_dw_times.append(t_expm_dw)
        diag_dw_times.append(t_diag_dw)

        expm_ent_times.append(t_expm_ent)
        diag_ent_times.append(t_diag_ent)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    """
    
    """

    axes[0].plot(L_vals, expm_fid_times, marker='o', label='expm_multiply')
    axes[0].plot(L_vals, diag_fid_times, marker='s', label='direct diagonalization')
    axes[0].set_yscale("log")
    axes[0].set_xlabel("L")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("Fidelity runtime")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(L_vals, expm_dw_times, marker='o', label='expm_multiply')
    axes[1].plot(L_vals, diag_dw_times, marker='s', label='direct diagonalization')
    axes[1].set_yscale("log")
    axes[1].set_xlabel("L")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_title("Domain-wall runtime")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(L_vals, expm_ent_times, marker='o', label='expm_multiply')
    axes[2].plot(L_vals, diag_ent_times, marker='s', label='direct diagonalization')
    axes[2].set_yscale("log")
    axes[2].set_xlabel("L")
    axes[2].set_ylabel("Runtime (s)")
    axes[2].set_title("Entanglement runtime")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("PXP: runtime comparison, expm_multiply vs direct diagonalization")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_observables_vs_time(L=12, tmax=30, nt=200, method="expm")
    plot_benchmarks()

'''
def gen_basis(L):
    out = []

    def rec(i, prev, s):
        # i = current site
        # prev = whether previous si
        te had a 1
        # s = integer representing configuration

        if i == L:
            out.append(s)
            return

        rec(i+1, 0, s)
        # Place 0 → always allowed

        if not prev:
            rec(i+1, 1, s | (1 << i))
        # Place 1 only if previous ≠ 1
        # Enforces constraint:
        # n_i n_{i+1} = 0

    rec(0, 0, 0)
    return out

# Hilbert space size scales as:
# dim ~ F_{L+2} (Fibonacci)

def basis_maps(L):
    b = gen_basis(L)
    return b, {s: i for i, s in enumerate(b)}
    # Maps state → matrix index
'''