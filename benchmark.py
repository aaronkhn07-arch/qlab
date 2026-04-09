import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


# Bit utilities

def bit_at(s, i):
    return (s >> i) & 1

def flip_bit(s, i):
    return s ^ (1 << i)

def z2_index(L):
    return sum((1 << i) for i in range(0, L, 2))


# Fibonacci basis generator (optimized)

def generate_valid_basis_fibonacci(L):
    basis = []

    def build(pos, prev_one, state):
        if pos == L:
            basis.append(state)
            return
        # place 0
        build(pos + 1, False, state)
        # place 1 only if previous wasn't 1
        if not prev_one:
            build(pos + 1, True, state | (1 << pos))

    build(0, False, 0)
    return basis

def build_basis_maps(L):
    basis = generate_valid_basis_fibonacci(L)
    index_map = {s: i for i, s in enumerate(basis)}
    return basis, index_map


# Hamiltonian

def build_pxp_hamiltonian(basis, index_map, L):
    rows, cols, data = [], [], []

    for i, s in enumerate(basis):
        for site in range(L):
            left = bit_at(s, site - 1) if site > 0 else 0
            right = bit_at(s, site + 1) if site < L - 1 else 0
            current = bit_at(s, site)

            if current == 1:
                s_new = flip_bit(s, site)
                j = index_map[s_new]
                rows.append(i); cols.append(j); data.append(1.0)
            elif left == 0 and right == 0:
                s_new = flip_bit(s, site)
                j = index_map[s_new]
                rows.append(i); cols.append(j); data.append(1.0)

    dim = len(basis)
    return csr_matrix((data, (rows, cols)), shape=(dim, dim))


# Observables

def z2_state(basis, index_map, L):
    psi = np.zeros(len(basis), dtype=complex)
    psi[index_map[z2_index(L)]] = 1.0
    return psi

def domain_wall_diag(basis, L):
    vals = []
    for s in basis:
        count = 0
        for i in range(L - 1):
            if bit_at(s, i) != bit_at(s, i + 1):
                count += 1
        vals.append(count / (L - 1))
    return np.array(vals)


# Entanglement helper

def prepare_entropy_helper(basis, L):
    L_half = L // 2
    left_map = {}
    right_map = {}
    left_idx = []
    right_idx = []

    for s in basis:
        left = s & ((1 << L_half) - 1)
        right = s >> L_half

        if left not in left_map:
            left_map[left] = len(left_map)
        if right not in right_map:
            right_map[right] = len(right_map)

        left_idx.append(left_map[left])
        right_idx.append(right_map[right])

    return np.array(left_idx), np.array(right_idx), len(left_map), len(right_map)

def entanglement_entropy(state, left_idx, right_idx, dimL, dimR):
    M = np.zeros((dimL, dimR), dtype=complex)
    for i, amp in enumerate(state):
        M[left_idx[i], right_idx[i]] += amp

    s = np.linalg.svd(M, compute_uv=False)
    p = s**2
    p = p[p > 1e-12]
    return -np.sum(p * np.log2(p))


# Time evolution

def evolve_expm(H, psi0, times):
    return np.array([expm_multiply(-1j * H * t, psi0) for t in times])


# Benchmark run

def run(L=20):
    print(f"\nRunning L={L}")

    t0 = time.time()
    basis, index_map = build_basis_maps(L)
    print("Basis size:", len(basis), " time:", time.time() - t0)

    t0 = time.time()
    H = build_pxp_hamiltonian(basis, index_map, L)
    print("H build time:", time.time() - t0, " nnz:", H.nnz)

    psi0 = z2_state(basis, index_map, L)
    dw = domain_wall_diag(basis, L)

    left_idx, right_idx, dimL, dimR = prepare_entropy_helper(basis, L)

    times = np.linspace(0, 10, 20)

    # Evolution
    t0 = time.time()
    states = evolve_expm(H, psi0, times)
    print("Evolution time:", time.time() - t0)

    # Fidelity
    fid = np.abs(states @ psi0.conj())**2

    # Domain walls
    dw_vals = [np.real(np.vdot(s, dw * s)) for s in states]

    # Entanglement
    ent = [entanglement_entropy(s, left_idx, right_idx, dimL, dimR) for s in states]

    print("Done.")
    return fid, dw_vals, ent


# Main

if __name__ == "__main__":
    run(20)