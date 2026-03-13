import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, LinearOperator

def entropy(psi, n_A, N):
    dim_a = 2**n_A
    dim_b = 2**(N - n_A)
    M = psi.reshape((dim_a, dim_b))
    s = np.linalg.svd(M, compute_uv=False)  # singular values
    t = s*s
    return -np.sum(t * np.log2(t))

# def pxp(N):
#     I = np.eye(2, dtype=np. complex128) # identity
#     X = np.array([[0, 1], [1, 0]], dtype=np.complex128) # pauli x
#     P = np.array([[1, 0], [0, 0]], dtype=np.complex128) # projector
#     d = 2**N
#     hamiltonian = np.zeros((d, d), dtype=np.complex128)
#     for i in range(N):
#         operators = [I] * N
#         operators[i] = X # flip the site of interest
        
#         if i-1 >= 0:
#             operators[i-1] = P
#         if i+1 < N:
#             operators[i+1] = P
        
#         out = operators[0] # start with identity and use the Kronecker product to add in the other states
#         for operator in operators[1:]:
#             out = np.kron(out, operator)
#         hamiltonian += out
#     return 

def pxp_bitwise(N):
    dim = 1 << L

def make_ising_linop(L, J=1.0, hx=1.0, hz=0.5, dtype=np.complex128):
    dim = 1 << L

    # Precompute diagonal energies for every computational basis state
    diagE = np.empty(dim, dtype=np.float64)

    # z_i = +1 if bit i == 0, else -1 (convention consistent with sz diag [1, -1])
    # i = 0 is least-significant bit site.
    for s in range(dim):
        # local z's
        z = np.empty(L, dtype=np.int8)
        for i in range(L):
            z[i] = 1 if ((s >> i) & 1) == 0 else -1

        e = 0.0
        if hz != 0.0:
            e += hz * np.sum(z)
        if J != 0.0:
            e += J * np.sum(z[:-1] * z[1:])
        diagE[s] = e

    # Precompute flip indices for hx term
    flips = [np.arange(dim, dtype=np.int64) ^ (1 << i) for i in range(L)]

    def matvec(v):
        v = v.astype(dtype, copy=False)
        out = (diagE * v).astype(dtype, copy=False)

        if hx != 0.0:
            # For each site i: (σ^x_i v)[s] = v[s ^ (1<<i)]
            # So contribution to out is hx * v[flip]
            for i in range(L):
                out += hx * v[flips[i]]
        return out

    return LinearOperator((dim, dim), matvec=matvec, dtype=dtype)

def ground_state(L, J=1.0, hx=1.0, hz=0.5):
    H = make_ising_linop(L, J=J, hx=hx, hz=hz)
    # k=1, smallest algebraic eigenvalue
    vals, vecs = eigsh(H, k=1, which="SA")
    psi0 = vecs[:, 0]
    # normalize (eigsh should already, but cheap safety)
    psi0 /= np.linalg.norm(psi0)
    return vals[0], psi0

def computeIsing_fast(N, Nsamples, J=1.0, hx=1.0, hz=0.5, Nmin=6):
    # Nsamples is ignored unless you introduce randomness; kept for API compatibility
    sizes = list(range(Nmin, N + 1))
    curves = []

    for L in sizes:
        E0, psi0 = ground_state(L, J=J, hx=hx, hz=hz)

        S = np.zeros(L + 1)
        # endpoints are exactly 0; and S(nA)=S(L-nA), so compute half
        S[0] = 0.0
        S[L] = 0.0
        for nA in range(1, L // 2 + 1):
            val = entropy(psi0, nA, L)
            S[nA] = val
            S[L - nA] = val

        curves.append((L, E0, S))

    plt.figure(figsize=(8, 8))
    for L, E0, S in curves:
        x = np.arange(L + 1) / L
        plt.plot(x, S / np.log(2), marker="o", linestyle="-", label=f"N={L}")

    plt.xlabel(r"$n_A/N$")
    plt.ylabel(r"$S(n_A)$ (bits)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example:
computeIsing_fast(15, 20, J=1.0, hx=1.0, hz=0.5, Nmin=6)
