# gmres.py
"""Volume 1: GMRES.
Patricia D. Bata
BUDS Program
11/07/2019
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy.sparse import linalg as spla
import time

#%%
# Problems 1 and 2.
def gmres(A, b, x0, k=100, tol=1e-8, plot=False):
    """Calculate approximate solution of Ax=b using the GMRES algorithm.

    Parameters:
        A ((m,m) ndarray): A square matrix.
        b ((m,) ndarray): A 1-D array of length m.
        x0 ((m,) ndarray): The initial guess for the solution to Ax=b.
        k (int): Maximum number of iterations of the GMRES algorithm.
        tol (float): Stopping criterion for size of residual.
        plot (bool): Whether or not to plot convergence (Problem 2).

    Returns:
        ((m,) ndarray): Approximate solution to Ax=b.
        res (float): Residual of the solution.
    """
    # Set up all empty vectors/matrices
    Q = np.empty((len(b), k+1))
    H = np.zeros((k+1, k))
    resid = np.empty(k)
    K = np.empty(k)
    
    #Set initial values
    r0 = b - A@x0
    b0 = la.norm(r0, 2)
    Q[:,0] = r0/b0
    be = np.append(b0, np.zeros(len(b)))

    # Iteration step
    for j in range(0, k):
        Q[:,j+1] = A@Q[:,j]
        for i in range(0,j+1):
            H[i,j] = np.inner(Q[:,i], Q[:,j+1])
            Q[:,j+1] -= H[i,j] * Q[:,i]
        H[j+1,j] = la.norm(Q[:,j+1], 2)
        if np.abs(H[j+1,j]) > tol:
            Q[:,j+1] /= H[j+1,j]
        a = H[:j+2,:j+1]
        b = be[:j+2]
        y, res = la.lstsq(a,b)[:2]
        res = np.sqrt(res)
        resid[j] = res
        if res < tol:
            resid = resid[:j+1]
            break 
    
    # Problem 2: Plotting
    if plot == True:
        fig, axes = plt.subplots(1, 2)
        # Plot 1 (Eigenvalues on complex plane)
        axes[0].scatter(la.eig(A)[0].real,la.eig(A)[0].imag)
        axes[0].set_title("Eigenvalues of A")
        axes[0].set_xlabel("Real component")
        axes[0].set_ylabel("Imaginary component")
        # Plot 2 (Residual vs iteration)
        K = np.linspace(0,j,j+1)
        axes[1].semilogy(K, resid)
        axes[1].set_title("Residuals of iterations")
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("Residuals")
        plt.tight_layout()
        plt.show()
        
    return Q[:,:(j+1)]@y + x0, res

#%%
# Problem 3
def prob3(m=200):
    """For n=-4,-2,0,2,4 create a matrix A= n*I + P where I is the mxm
    identity, and P is an mxm matrix with entries drawn from a normal
    distribution with mean 0 and standard deviation 1/(2*sqrt(m)).
    For each of the given values of n call gmres() with A, a vector of ones called b, an initial guess x0=0, and plot=True

    Parameters:
        m (int): Size of the matrix A.
    """
    x0 = np.zeros(m)
    b = np.ones(m)
    P = np.random.normal(loc = 0, scale = 1/(2*np.sqrt(m)), size = (m,m))
    for n in [-4,-2,0,2,4]:
        A = n*np.eye(m) + P
        gmres(A, b, x0, k=100, tol=1e-8, plot=True)
        
#%%
# Problem 4
def gmres_k(A, b, x0, k=5, tol=1E-8, restarts=50):
    """Implement the GMRES algorithm with restarts. Terminate the algorithm
    when the size of the residual is less than tol or when the maximum number
    of restarts has been reached.

    Parameters:
        A ((m,m) ndarray): A square matrix.
        b ((m,) ndarray): A 1-D array of length m.
        x0 ((m,) ndarray): The initial guess for the solution to Ax=b.
        k (int): Maximum number of iterations of the GMRES algorithm.
        tol (float): Stopping criterion for size of residual.
        restarts (int): Maximum number of restarts. Defaults to 50.

    Returns:
        ((m,) ndarray): Approximate solution to Ax=b.
        res (float): Residual of the solution.
    """
    # Restarts loop, stores new x0 if res < tol
    for r in range(0, restarts+1):
        x0, res = gmres(A, b, x0, k, tol)
        if res < tol:
            break
    return x0, res

#%%
# Problem 5
def time_gmres(m=200):
    """Using the same matrices as in problem 2, plot the time required to
    complete gmres(), gmres_k(), and scipy.sparse.linalg.gmres() with
    restarts. Plot the values of m against the times.
    """
    # Define empty list and sizes of the matrices
    prob1 = []
    prob4 = []
    scipy = []
    x = np.arange(25,201,25)
    
    # Self defined timing function
    def timed(f,*args, **kwargs):
        time1 = time.time()
        f(*args, **kwargs)
        time2 = time.time()
        return time2 - time1
    
    # Timing the three functions 
    for m in x:
        P = np.random.normal(loc = 0, scale = 1/(2*np.sqrt(m)), size = (m,m))
        x0 = np.zeros(m)
        b = np.ones(m)
        prob1.append(timed(gmres, P, b, x0, k=100, tol=1e-8))
        prob4.append(timed(gmres_k, P, b, x0, k=5, tol=1E-8, restarts=50))
        scipy.append(timed(spla.gmres, P, b, restart = 1000))
    
    # Plot all of the times
    for j in [prob1, prob4, scipy]:
        plt.plot(x, j, '-o')
    plt.xlabel("m")
    plt.ylabel("Runtime (s)")
    plt.title("Runtimes of gmres, gmres_k, and spla.gmres")
    plt.legend(labels = ["gmres", "gmres_k", "spla.gmres"])
    plt.show()
