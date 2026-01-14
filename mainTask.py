import numpy as np
import matplotlib.pyplot as plt
import math

def random_state (N):
    #returns an array whose components correspond to each basis via gaussian random numbers
    psi = np.random.randn(2**N) + 1j*np.random.randn(2**N)
    psi /= np.linalg.norm(psi) 
    return psi

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


compute (14, 50)