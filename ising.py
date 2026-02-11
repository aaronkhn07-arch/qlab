import numpy as np
import matplotlib.pyplot as plt

def ising(N, J=1.0, hx=1.0, hz=0.5):
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    ident = np.eye(2)

    def build_op(ops_by_site):
        op = np.array([[1.0]])
        for site in range(N):
            op = np.kron(op, ops_by_site.get(site, ident))
        return op

    dim = 2**N
    h = np.zeros((dim, dim))

    for i in range(N - 1):
        h += J * build_op({i: sz, i + 1: sz})

    for i in range(N):
        if hx != 0.0:
            h += hx * build_op({i: sx})
        if hz != 0.0:
            h += hz * build_op({i: sz})

    return h

def entropy(psi, n_A, N):
    dim_a = 2**n_A
    dim_b = 2**(N - n_A)
    matrixPsi = psi.reshape((dim_a, dim_b))
    svals = np.linalg.svd(matrixPsi, compute_uv=False)
    svals = svals[svals > 1e-12]
    lambdas = svals**2
    return -np.sum(lambdas * np.log2(lambdas))

def pageApproximation(nA, N):
    m = 2**nA
    n = 2**(N - nA)
    if m > n:
        m, n = n, m
    approximation = sum(1 / k for k in range(n + 1, n * m + 1)) - (m - 1) / (2 * n)
    return approximation / np.log(2)

def volumelaw(
    Ns=(8, 9, 10, 11, 12, 13, 14),
    fractions=(0.25, 0.33, 0.5),
    Nsamples=30,
    energy_window=(0.45, 0.55),
    J=1.0, hx=1.0, hz=0.5
):
    results = {f: {"N": [], "nA": [], "S": [], "S_per_site": [], "Page": []} for f in fractions}

    for N in Ns:
        H = ising(N, J=J, hx=hx, hz=hz)
        evals, evecs = np.linalg.eigh(H)

        Emin, Emax = float(evals[0]), float(evals[-1])
        denom = (Emax - Emin) if (Emax > Emin) else 1.0
        e_norm = (evals - Emin) / denom

        lo, hi = energy_window
        idx_pool = np.where((e_norm >= lo) & (e_norm <= hi))[0]

        # Fallback if the window is too narrow at small N
        if len(idx_pool) == 0:
            idx_pool = np.arange(1, len(evals) - 1)

        for f in fractions:
            nA = int(np.floor(f * N))
            nA = max(1, min(nA, N - 1))

            # Sample eigenstates from the chosen energy window
            picks = np.random.choice(idx_pool, size=min(Nsamples, len(idx_pool)), replace=(len(idx_pool) < Nsamples))

            S_accum = 0.0
            for k in picks:
                psi = evecs[:, int(k)]
                S_accum += entropy(psi, nA, N)
            S_mean = S_accum / len(picks)

            results[f]["N"].append(N)
            results[f]["nA"].append(nA)
            results[f]["S"].append(S_mean)
            results[f]["S_per_site"].append(S_mean / nA)

            # Page approximation for comparison (random states at the same bipartition size)
            results[f]["Page"].append(pageApproximation(nA, N))

    # S(fN) v N and a best fit line in N. volume law => linear in N
    plt.figure(figsize=(8, 6))
    for f in fractions:
        N_arr = np.array(results[f]["N"], dtype=float)
        S_arr = np.array(results[f]["S"], dtype=float)
        plt.plot(N_arr, S_arr, marker="o", linestyle="-", label=f"Ising, nA=fN, f={f}")

        # Linear fit S = a N + b
        if len(N_arr) >= 2:
            a, b = np.polyfit(N_arr, S_arr, 1)
            plt.plot(N_arr, a * N_arr + b, linestyle="--", label=f"fit: a={a:.3f}")

    plt.xlabel("N")
    plt.ylabel("S(nA)")
    plt.title("S(fN) grows linearly in N (=> Volume Law)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # S(nA)/nA v N
    plt.figure(figsize=(8, 6))
    for f in fractions:
        N_arr = np.array(results[f]["N"], dtype=float)
        Sper_arr = np.array(results[f]["S_per_site"], dtype=float)
        plt.plot(N_arr, Sper_arr, marker="o", linestyle="-", label=f"Ising, S/nA, f={f}")

    plt.xlabel("N")
    plt.ylabel("S(nA)/nA")
    plt.title("S(nA)/nA stays bounded away from 0 (Volume Law)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # compare to page
    plt.figure(figsize=(8, 6))
    for f in fractions:
        N_arr = np.array(results[f]["N"], dtype=float)
        S_arr = np.array(results[f]["S"], dtype=float)
        P_arr = np.array(results[f]["Page"], dtype=float)
        plt.plot(N_arr, S_arr, marker="o", linestyle="-", label=f"Ising S, f={f}")
        plt.plot(N_arr, P_arr, marker="x", linestyle="--", label=f"Page approx, f={f}")

    plt.xlabel("N")
    plt.ylabel("S(nA)")
    plt.title("Ising eigenstates vs Page approximation (same bipartitions)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # (hx != 0 and hz != 0)
    volumelaw(
        Ns=(8, 9, 10, 11, 12, 13, 14),
        fractions=(0.25, 0.33, 0.5),
        Nsamples=40,
        energy_window=(0.45, 0.55),
        J=1.0, hx=1.0, hz=0.5
    )