import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, LinearOperator

def valid_states(N):
    valid = []
    for s in range(1 << N):
        if not (s & (s << 1)): # check for adjacent 1s
            valid.append(s)
    return valid

def pxp_fib_sparse(N):
    valid = valid_states(N)
    dim = len(valid)
    h = np.zeros((dim, dim), dtype=np.float64)
    index_map = {s: i for i, s in enumerate(valid)} # bit int to row/col index

    for i, s in enumerate(valid):
        for j in range(N):
            if (s & (1 << j)) == 0: # site j is 0, can flip to 1
                left_ok = (j == 0) or ((s & (1 << (j - 1))) == 0) # check left neighbor
                right_ok = (j == N - 1) or ((s & (1 << (j + 1))) == 0) # check right neighbor
                if left_ok and right_ok:
                    sf = s | (1 << j) # flip the jth bit to 1
                    if sf in index_map:
                        h[index_map[sf], i] += 1.0 # hamiltonian[row, col]++
    return h

def z2_fib(N, id):
    s = sum(1 << i for i in range(0, N, 2)) # 1010...10
    psi = np.zeros(len(id), dtype=np.complex128)
    psi[id[s]] = 1.0
    return psi

def domain_wall_fib(L, valid):
    dw = np.zeros(len(valid))
    for i, s in enumerate(valid):
        total = sum (0.5 * (1.0 - (1.0 if ((s >> j) & 1) == 0 else -1.0) * (1.0 if ((s >> (j + 1)) & 1) == 0 else -1.0)) for j in range(L - 1))
        dw[i] = total / (L - 1)
    return dw

def embed(psi_fib, valid, N):
    psi_full = np.zeros(1<<N, dtype=np.complex128)
    for i, s in enumerate(valid):
        psi_full[s] = psi_fib[i]
    return psi_full


def entropy(psi, n_A, N):
    dim_a = 2**n_A
    dim_b = 2**(N - n_A)
    M = psi.reshape((dim_a, dim_b))
    s = np.linalg.svd(M, compute_uv=False)  # singular values
    t = s*s
    return -np.sum(t * np.log2(t))

def pxp(N):
    I = np.eye(2, dtype=np. complex128) # identity
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128) # pauli x
    P = np.array([[1, 0], [0, 0]], dtype=np.complex128) # projector
    d = 2**N
    hamiltonian = np.zeros((d, d), dtype=np.complex128)
    for i in range(N):
        operators = [I] * N
        operators[i] = X # flip the site of interest
        
        if i-1 >= 0:
            operators[i-1] = P
        if i+1 < N:
            operators[i+1] = P
        
        out = operators[0] # start with identity and use the Kronecker product to add in the other states
        for operator in operators[1:]:
            out = np.kron(out, operator)
        hamiltonian += out
    return hamiltonian

def pxp_bitwise(N):
    dim = 1 << N
    h = np.zeros((dim, dim), dtype=np.float64)
    for s in range(dim):
        for i in range(N):
            left_ok = (i == 0) or (((s >> (i - 1)) & 1) == 0) # check left neighbor
            right_ok = (i == N - 1) or (((s >> (i + 1)) & 1) == 0) # check right neighbor
            if left_ok and right_ok:
                sf = s ^ (1 << i) # flip the ith bit
                h[sf, s] += 1.0
    return h

def mf_ising(
    L: int, J: float = 1.0, hx: float = 1.0, hz: float = 0.5
) -> np.ndarray:
    """Mixed-field Ising H = J sum_i Z_i Z_{i+1} + hx sum_i X_i + hz sum_i Z_i."""
    dim = 1 << L
    h = np.zeros((dim, dim), dtype=np.float64)

    for s in range(dim):
        e_diag = 0.0
        for i in range(L - 1):
            zi = 1.0 if ((s >> i) & 1) == 0 else -1.0
            zip1 = 1.0 if ((s >> (i + 1)) & 1) == 0 else -1.0
            e_diag += J * zi * zip1
        for i in range(L):
            zi = 1.0 if ((s >> i) & 1) == 0 else -1.0
            e_diag += hz * zi
            sf = s ^ (1 << i)
            h[sf, s] += hx
        h[s, s] += e_diag
    return h

def z2_state(L: int) -> np.ndarray:
    """|Z2> = |1010...10> for even L."""
    dim = 1 << L
    state_index = 0
    for i in range(L):
        if i % 2 == 0:
            state_index |= 1 << i # set the ith bit to 1 for even i
    psi = np.zeros(dim, dtype=np.complex128)
    psi[state_index] = 1.0
    return psi

def precompute_diag(L: int) -> np.ndarray:
    """Diagonal entries of mean domain-wall density operator."""
    dim = 1 << L
    dw = np.zeros(dim, dtype=np.float64)
    for s in range(dim):
        total = 0.0
        for i in range(L - 1):
            zi = 1.0 if ((s >> i) & 1) == 0 else -1.0 # +1 for spin up, -1 for spin down
            zip1 = 1.0 if ((s >> (i + 1)) & 1) == 0 else -1.0 #
            total += 0.5 * (1.0 - zi * zip1)
        dw[s] = total / (L - 1)
    return dw

def half_chain_entropy(psi: np.ndarray, L: int, eps: float = 1e-14) -> float:
    n_a = L // 2
    mat = psi.reshape((1 << n_a, 1 << (L - n_a)))
    svals = np.linalg.svd(mat, compute_uv=False)
    probs = svals * svals
    probs = probs[probs > eps]
    return float(-np.sum(probs * np.log(probs)))

def evolve_and_measure(
    hamiltonian: np.ndarray, psi0: np.ndarray, times: np.ndarray, domain_wall_diag: np.ndarray, L: int
) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(hamiltonian)
    coeffs0 = evecs.conj().T @ psi0

    dw_curve = np.empty_like(times, dtype=np.float64)
    ent_curve = np.empty_like(times, dtype=np.float64)

    for ti, t in enumerate(times):
        phase = np.exp(-1j * evals * t)
        psi_t = evecs @ (phase * coeffs0) # this line computes the time-evolved state |psi(t)> = sum_n e^{-i E_n t} <E_n|psi0> |E_n>
        probs = np.abs(psi_t) ** 2
        dw_curve[ti] = float(np.dot(probs, domain_wall_diag))
        ent_curve[ti] = half_chain_entropy(psi_t, L)
    return dw_curve, ent_curve

def evolve_and_measure_fib(
    hamiltonian: np.ndarray, psi0: np.ndarray, times: np.ndarray, domain_wall_diag: np.ndarray,
    valid: list, L: int
) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(hamiltonian)
    coeffs0 = evecs.conj().T @ psi0

    dw_curve = np.empty_like(times, dtype=np.float64)
    ent_curve = np.empty_like(times, dtype=np.float64)

    for ti, t in enumerate(times):
        phase = np.exp(-1j * evals * t)
        psi_t = evecs @ (phase * coeffs0)
        probs = np.abs(psi_t) ** 2
        dw_curve[ti] = float(np.dot(probs, domain_wall_diag))
        ent_curve[ti] = half_chain_entropy(embed(psi_t, valid, L), L)
    return dw_curve, ent_curve


def quench(
    L: int = 12, t_max: float = 30.0, n_times: int = 1201, J: float = 1.0, hx: float = 1.0, hz: float = 0.5
) -> None:
    times = np.linspace(0.0, t_max, n_times)

    valid = valid_states(L)
    index_map = {s: i for i, s in enumerate(valid)}

    h_pxp = pxp_fib_sparse(L)
    psi0_pxp = z2_fib(L, index_map)
    dw_diag_pxp = domain_wall_fib(L, valid)

    psi0_ising = z2_state(L)
    dw_diag_ising = precompute_diag(L)
    h_ising = mf_ising(L, J=J, hx=hx, hz=hz)

    dw_pxp, s_pxp = evolve_and_measure_fib(h_pxp, psi0_pxp, times, dw_diag_pxp, valid, L)
    dw_ising, s_ising = evolve_and_measure(h_ising, psi0_ising, times, dw_diag_ising, L)

    np.savez(
        "quench_L12_data.npz",
        times=times,
        domain_wall_pxp=dw_pxp,
        entanglement_pxp=s_pxp,
        domain_wall_ising=dw_ising,
        entanglement_ising=s_ising,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(times, dw_pxp, label="PXP", lw=2)
    axes[0].plot(times, dw_ising, label="Mixed-field Ising", lw=2)
    axes[0].set_ylabel("Domain wall density")
    axes[0].set_title(f"Quench from |Z2>, L={L}, t in [0, {t_max}]")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(times, s_pxp, label="PXP", lw=2)
    axes[1].plot(times, s_ising, label="Mixed-field Ising", lw=2)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Half-chain entanglement entropy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("quench_L12_comparison.png", dpi=180)
    plt.show()

if __name__ == "__main__":
    quench(L=12, t_max=30.0, n_times=1201, J=1.0, hx=1.0, hz=0.5)
