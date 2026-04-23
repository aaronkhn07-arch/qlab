"""
Compare binary strings of length L following the open bounary conditions;

1. Tree search with pruning visits all F(L+4) nodes (valid prefixes of every length <= L)

2. Divide and conquer with dictionaries recursively splits the chain in half, generates substrings for each half, and combines using four diciontaries indexed by left and right bits. F(L+2) at the top level

Both produce F(L+2) valid states
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# PRecompute the valid combinations for divide and conquer
# 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
# for (leftmost_bit, rightmost_bit) of a substring

VALID_COMBOS = []
for ll in (0, 1): # leftmost bit of left substring
    for lr in (0, 1): # rightmost bit of left substring
        for rl in (0, 1): # leftmost bit of right substring
            if lr == 1 and rl == 1:
                continue # boundary violation
            for rr in (0, 1): # rightmost bit of right substring
                VALID_COMBOS.append((ll * 2 + lr, # left dict index
                                     rl * 2 + rr, # right dict index
                                     ll * 2 + rr)) # result dict index


# Tree search
def tree_search(L):
    out = []
    def rec(i, prev, s):
        if i == L:
            out.append(s)
            return
        rec(i + 1, 0, s)
        if not prev:
            rec(i + 1, 1, s | (1 << i))
    rec(0, 0, 0)
    return out

# Divide and conquer
def _dnc_dicts(L):
    # Return (D00, D01, D10, D11) for valid strings
    if L == 1:
        return ([0], [], [], [1])

    llen = L // 2
    rlen = L - llen

    LD = _dnc_dicts(llen)
    RD = _dnc_dicts(rlen)

    result = [[], [], [], []]

    for l_idx, r_idx, res_idx in VALID_COMBOS:
        left_list = LD[l_idx]
        right_list = RD[r_idx]
        if not left_list or not right_list:
            continue
        right_shifted = [rs << llen for rs in right_list]
        result[res_idx].extend(
            ls | rs_s for ls in left_list for rs_s in right_shifted
        )

    return tuple(result)


def divide_and_conquer(L):
    D = _dnc_dicts(L)
    return D[0] + D[1] + D[2] + D[3]


# Verification
def verify(L_max=20):
    # Check that both methods match for L = 1 to L_max
    for L in range(1, L_max + 1):
        s1 = sorted(tree_search(L))
        s2 = sorted(divide_and_conquer(L))
        assert s1 == s2, f"Mismatch at L={L}: {len(s1)} vs {len(s2)}"
    print(f"Verification passed for L = 1 .. {L_max}")


# Benhchmark
def benchmark(L_values, n_trials=5):
    results = {'L': [], 'bfs_time': [], 'dnc_time': [],
               'ratio': [], 'n_states': []}

    for L in L_values:
        bfs_times, dnc_times = [], []

        for _ in range(n_trials):
            t0 = time.perf_counter()
            s1 = tree_search(L)
            bfs_times.append(time.perf_counter() - t0)

        for _ in range(n_trials):
            t0 = time.perf_counter()
            s2 = divide_and_conquer(L)
            dnc_times.append(time.perf_counter() - t0)

        t_bfs = min(bfs_times)
        t_dnc = min(dnc_times)
        ratio = t_bfs / t_dnc if t_dnc > 0 else float('inf')

        results['L'].append(L)
        results['bfs_time'].append(t_bfs)
        results['dnc_time'].append(t_dnc)
        results['ratio'].append(ratio)
        results['n_states'].append(len(s1))

        print(f"L={L:3d}  |basis|={len(s1):>10d}  "
              f"tree={t_bfs:.5f}s  D&C={t_dnc:.5f}s  ratio={ratio:.2f}")

    return results


# Plotting
def plot_results(results):
    L = np.array(results['L'], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Absolute runtimes (log scale) ---
    ax = axes[0]
    ax.semilogy(L, results['bfs_time'], 'o-', label='Tree search')
    ax.semilogy(L, results['dnc_time'], 's-', label='Divide & conquer')
    ax.set_xlabel('System size $L$')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Ratio with linear fit ---
    ax = axes[1]
    ratio = np.array(results['ratio'])
    ax.plot(L, ratio, 'o-', color='tab:green')
    coeffs = np.polyfit(L, ratio, 1)
    L_fit = np.linspace(L[0], L[-1], 100)
    ax.plot(L_fit, np.polyval(coeffs, L_fit), '--', color='gray',
            label=f'linear fit: {coeffs[0]:.3f}$L$ + {coeffs[1]:.2f}')
    ax.set_xlabel('System size $L$')
    ax.set_ylabel('Tree time / D&C time')
    ax.set_title('Runtime ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Hilbert space dimension ---
    ax = axes[2]
    ax.semilogy(L, results['n_states'], 'o-', color='tab:red')
    ax.set_xlabel('System size $L$')
    ax.set_ylabel('Number of valid states')
    ax.set_title('Hilbert space dimension $F(L+2)$')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('blockade_basis_benchmark.png', dpi=150)
    plt.show()
    print("Saved blockade_basis_benchmark.png")


if __name__ == '__main__':
    verify()

    L_values = list(range(4, 31, 2))
    print("\nBenchmarking (5 trials each, reporting minimum time):\n")
    results = benchmark(L_values)

    print("\nPlotting...")
    plot_results(results)
