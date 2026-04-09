import numpy as np
import matplotlib.pyplot as plt
import math

def random_state (N):
    #returns an array whose components correspond to each basis via gaussian random numbers
    psi = np.random.randn(2**N) + 1j*np.random.randn(2**N)
    psi /= np.linalg.norm(psi) 
    return psi

def guassianEnsemble (N): # returns a gue matrix m
    m = np.zeros((N, N), dtype=complex)
    r = np.random.randn(N)
    for i in range(N): # diag
        m[i][i] = r[i]
    for i in range(N): # off diag
        for j in range(i):
            n = np.random.normal() + 1j * np.random.normal()
            m[i][j] = n
            m[j][i] = np.conj(n)
    return m

def ising (N, J=1.0, hx=1.0, hz=0.5): #returns a mixed-field Ising Hamiltonian (open chain)
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


def gueEigenstate(m): # return a random eigenstate of a gue matrix m, excluding the first and last eigenstates
    eigvals, eigvecs = np.linalg.eigh(m)
    index = np.random.randint(1, len(eigvals)-1)
    return eigvecs[:, index]

def entropy (psi, n_A, N): #returns entanglement entropy for psi
    dim_a = 2**(n_A)
    dim_b = 2**(N - n_A)

    matrixPsi = psi.reshape ((dim_a, dim_b))

    svals = np.linalg.svd(matrixPsi, compute_uv=False) #we heard that svd is more optimized than eigvalsh. So the squares here are the eigenvalues we want
    svals = svals[svals > 1e-12] #to avoid 0 being viewed as nonzero so we use lazy boolean to convert those that are sufficiently small into 0

    lambdas = svals**2 #it looks like the mistake was that we only squared the outer lambdas in our original code. We forgot to square svals for the argument of log2
    return -np.sum(lambdas * np.log2(lambdas)) #entanglement formula. We also need to square svals

def pageApproximation (nA, N):
    m = 2**nA
    n = 2**(N - nA)

    if m > n: m, n = n, m

    approximation = sum (1/k for k in range (n + 1, n*m + 1)) - (m - 1)/(2*n)

    return approximation / np.log(2)

def compute (N, Nsamples): #gives the plot
    Savg  = np.zeros (N + 1) #an array of the average entanglement entropies

    for sample in range (Nsamples): 
        psi = random_state (N)

        for nA in range (N + 1): Savg[nA] += entropy (psi, nA, N)

    Savg /= Nsamples

    #computing differences for Don Page's formula
    difference = np.zeros (N + 1)
    for nA in range (N + 1): difference[nA] = Savg[nA] - pageApproximation (nA, N)

    pageApp = np.zeros(N + 1)
    for i in range (N+1): pageApp[i] = pageApproximation (i, N)
    maxVal = np.zeros (N + 1)
    for i in range (N + 1):
        maxVal[i] = i if i <= N - i else N - i

    print(pageApp)
    plt.figure (figsize=(8, 8))
    plt.plot (Savg, marker = "o", linestyle = "-")
    plt.xlabel (r'nA')
    plt.ylabel (r'Difference')
    
    plt.grid(True)
    plt.show()

    nA_vals = np.arange (N + 1)
    plt.figure(figsize=(8, 8))
    plt.plot(nA_vals, Savg/np.log(2), marker = "o", linestyle = "-", color = 'red')
    plt.plot (nA_vals, pageApp/np.log(2), linestyle = '-', color = 'blue')
    plt.plot (nA_vals, maxVal/np.log(2), linestyle = '-', color = 'yellow')
    plt.xlabel (r'$n_A$ (number of qubits in subsystem A)')
    plt.ylabel (r'$\overline{S(n_A)}$ (average entanglement entropy)')
    
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(nA_vals, nA_vals, marker = "o", linestyle = "-", color = "red")
    plt.xlabel (r'$n_A$')
    plt.ylabel (r"$max (S(A))$")
    plt.title ("Maxmimum entanglement entropy to number of qubits in the subsystem")
    plt.grid(True)
    plt.show()

    plt.figure (figsize=(8, 8))
    plt.plot (nA_vals, pageApp, marker = "o", linestyle = "-")
    plt.xlabel (r'$n_A$')
    plt.ylabel ("Page approx.")
    plt.show()

def computeGUE(N, Nsamples):
    Savg  = np.zeros (2**N) #an array of the average entanglement entropies

    for sample in range (Nsamples): 
        gue = guassianEnsemble(2**N)
        psi = gueEigenstate(gue)

        for nA in range (2**N): Savg[nA] += entropy (psi, nA, int(math.log2(2**N)))

    Savg /= Nsamples

    pageApp = np.zeros(N + 1)
    for i in range (N+1): pageApp[i] = pageApproximation (i, N)
    maxVal = np.zeros (N + 1)
    for i in range (N + 1):
        maxVal[i] = i if i <= N - i else N - i
    

    plt.figure (figsize=(8, 8))
    plt.plot (Savg/np.log(2), marker = "o", linestyle = "-")
    plt.plot (pageApp/np.log(2), linestyle = '-', color = 'blue')
    plt.plot (maxVal/np.log(2), linestyle = '-', color = 'yellow')

    plt.xlabel (r'nA')
    plt.ylabel (r'Average Entanglement Entropy')
    
    plt.grid(True)
    plt.show()

def computeIsing(N, Nsamples, J=1.0, hx=1.0, hz=0.5):
    Savg = np.zeros(N + 1)

    for sample in range(Nsamples):
        h = ising(N, J=J, hx=hx, hz=hz)
        eigvals, eigvecs = np.linalg.eigh(h)
        index = np.random.randint(1, len(eigvals) - 1)
        psi = eigvecs[:, index]

        for nA in range(N + 1):
            Savg[nA] += entropy(psi, nA, N)

    Savg /= Nsamples

    pageApp = np.zeros(N + 1)
    for i in range(N + 1):
        pageApp[i] = pageApproximation(i, N)
    maxVal = np.zeros(N + 1)
    for i in range(N + 1):
        maxVal[i] = i if i <= N - i else N - i

    plt.figure(figsize=(8, 8))
    plt.plot(Savg / np.log(2), marker="o", linestyle="-")
    plt.plot(pageApp / np.log(2), linestyle="-", color="blue")
    plt.plot(maxVal / np.log(2), linestyle="-", color="yellow")

    plt.xlabel(r"nA")
    plt.ylabel(r"Average Entanglement Entropy")

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    computeGUE (14, 50)