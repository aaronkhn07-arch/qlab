# -----------------------
# Control threading (for consistent benchmarking)
# -----------------------
import os
os.environ["OMP_NUM_THREADS"] = "1"   # Force single-thread BLAS
os.environ["MKL_NUM_THREADS"] = "1"   # Avoid hidden parallel speedups

# -----------------------
# Imports
# -----------------------
import time                       # timing benchmarks
import numpy as np               # numerical arrays
import matplotlib.pyplot as plt  # plotting
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

# -----------------------
# Bit utilities
# -----------------------

def bit(s, i):
    return (s >> i) & 1
    # Extracts bit at site i from integer s
    # If s = binary configuration, this gives n_i ∈ {0,1}

def flip(s, i):
    return s ^ (1 << i)
    # Flips spin at site i:
    # s → s ⊕ (1 << i)

def z2(L):
    return sum(1 << i for i in range(0, L, 2))
    # Builds |Z2⟩ = |1010...⟩ state
    # i.e. spins on even sites = 1

# -----------------------
# Fibonacci basis (PXP constraint)
# -----------------------

def gen_basis(L):
    out = []

    def rec(i, prev, s):
        # i = current site
        # prev = whether previous site was 1
        # s = current bitstring

        if i == L:
            out.append(s)
            return

        rec(i+1, 0, s)
        # Place 0 → always allowed

        if not prev:
            rec(i+1, 1, s | (1<<i))
        # Place 1 only if previous ≠ 1
        # Enforces PXP constraint: n_i n_{i+1} = 0

    rec(0, 0, 0)
    return out

# This generates ONLY valid states → scales like Fibonacci F_{L+2}
# instead of 2^L

def basis_maps(L):
    b = gen_basis(L)
    return b, {s:i for i,s in enumerate(b)}
    # Maps each state → matrix index

# -----------------------
# Hamiltonian construction
# -----------------------

def build_H(basis, idx, L):
    r,c,d = [],[],[]

    for i,s in enumerate(basis):
        for j in range(L):

            l = bit(s,j-1) if j>0 else 0
            rgt = bit(s,j+1) if j<L-1 else 0
            cur = bit(s,j)

            # PXP Hamiltonian:
            # H = Σ_i P_{i-1} X_i P_{i+1}
            #
            # where:
            # P = |0><0|  (projector onto empty site)
            # X flips spin

            if cur==1:
                # X|1⟩ = |0⟩ → always allowed
                sn = flip(s,j)
                r.append(i); c.append(idx[sn]); d.append(1.0)

            elif l==0 and rgt==0:
                # Only flip 0→1 if neighbors are empty
                sn = flip(s,j)
                r.append(i); c.append(idx[sn]); d.append(1.0)

    n = len(basis)
    return csr_matrix((d,(r,c)), shape=(n,n))
    # Sparse Hamiltonian matrix

# -----------------------
# Observables
# -----------------------

def z2_state(idx,L,n):
    psi = np.zeros(n,complex)
    psi[idx[z2(L)]] = 1
    return psi
    # Initial state:
    # |ψ(0)⟩ = |Z2⟩

def domain_wall(basis,L):
    return np.array([
        sum(bit(s,i)!=bit(s,i+1) for i in range(L-1))/(L-1)
        for s in basis
    ])
    # Domain wall density:
    # D = (1/(L-1)) Σ_i (n_i XOR n_{i+1})

# -----------------------
# Entanglement entropy
# -----------------------

def prep_entropy(basis,L):
    h=L//2

    Lmap,Rmap={},{}
    Li,Ri=[],[]

    for s in basis:
        l = s & ((1<<h)-1)   # left half bits
        r = s >> h           # right half bits

        if l not in Lmap: Lmap[l]=len(Lmap)
        if r not in Rmap: Rmap[r]=len(Rmap)

        Li.append(Lmap[l])
        Ri.append(Rmap[r])

    return np.array(Li),np.array(Ri),len(Lmap),len(Rmap)

# Precomputes mapping:
# |ψ⟩ → matrix M_{αβ}

def entropy(psi,Li,Ri,dL,dR):
    M=np.zeros((dL,dR),complex)

    for i,a in enumerate(psi):
        M[Li[i],Ri[i]]+=a
    # Build wavefunction matrix

    s=np.linalg.svd(M,compute_uv=False)

    p=s*s
    p=p[p>1e-12]

    return -np.sum(p*np.log2(p))
    # Von Neumann entropy:
    # S = -Tr(ρ log₂ ρ)
    # where p = singular values²

# -----------------------
# Time evolution
# -----------------------

def evolve(H,psi,t):
    return np.array([expm_multiply(-1j*H*ti,psi) for ti in t])

# Computes:
# |ψ(t)⟩ = exp(-i H t) |ψ(0)⟩

# -----------------------
# Benchmarking
# -----------------------

def benchmark(Ls=[10,12,14,16,18,20]):
    basis_t, evolve_t, ent_t = [], [], []

    for L in Ls:
        print(f"\nL={L}")

        # --- Basis generation ---
        t0=time.time()
        basis, idx = basis_maps(L)
        tb=time.time()-t0
        basis_t.append(tb)

        # --- Hamiltonian ---
        H=build_H(basis,idx,L)

        psi0=z2_state(idx,L,len(basis))
        dw=domain_wall(basis,L)

        Li,Ri,dL,dR=prep_entropy(basis,L)

        tgrid=np.linspace(0,5,10)

        # --- Time evolution ---
        t0=time.time()
        states=evolve(H,psi0,tgrid)
        te=time.time()-t0
        evolve_t.append(te)

        # --- Entanglement ---
        t0=time.time()
        _=[entropy(s,Li,Ri,dL,dR) for s in states]
        tent=time.time()-t0
        ent_t.append(tent)

        print(f"basis={tb:.3f}s evolve={te:.3f}s ent={tent:.3f}s dim={len(basis)}")

    return Ls, basis_t, evolve_t, ent_t

# -----------------------
# Plotting
# -----------------------

def plot_results(Ls, b, e, ent):
    plt.figure()
    plt.plot(Ls,b,'o-',label="Basis (Fibonacci)")
    plt.plot(Ls,e,'o-',label="Evolution (expm)")
    plt.plot(Ls,ent,'o-',label="Entanglement")

    plt.xlabel("L")
    plt.ylabel("Time (s)")
    plt.title("PXP Scaling Benchmark")
    plt.legend()
    plt.grid()

    plt.savefig("pxp_scaling.png",dpi=150)

    # Relative cost plot
    plt.figure()
    plt.plot(Ls,np.array(ent)/np.array(e),'o-')

    plt.xlabel("L")
    plt.ylabel("Entanglement / Evolution time")
    plt.title("Relative Cost")
    plt.grid()

    plt.savefig("pxp_relative_cost.png",dpi=150)

# -----------------------
# Main
# -----------------------

if __name__=="__main__":
    Ls,b,e,ent=benchmark()
    plot_results(Ls,b,e,ent)