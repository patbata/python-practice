# interior_point_quadratic.py
"""Volume 2: Interior Point for Quadratic Programs.
Patricia D. Bata
BUDS Program
11/19/2019
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import spdiags
from matplotlib import pyplot as plt
from cvxopt import matrix, solvers
from mpl_toolkits.mplot3d import axes3d


#%%
def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Parameters:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    # Initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0

#%%
# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    #Store m, n of A
    m,n = A.shape
    
    #Counter check if input sizes are wrong
    assert len(b) == m, "A and b not aligned"
    assert Q.shape[0] == Q.shape[1] == len(c) == n, "Q, A and c not aligned"


    def F(xi, yi, mi):
        """The almost-linear function that accounts for the KKT conditions."""
        return np.hstack((np.dot(Q, xi) - np.dot(A.T, mi) + c,
                          np.dot(A, xi) - yi - b,
                          mi*yi))
    
    #Setup of initial DF
    DF = np.vstack((np.hstack((Q, np.zeros((n,m)), -A.T)),
                    np.hstack((A, -np.eye(m), np.zeros((m,m)))),
                    np.zeros((m,2*m+n))))

    # Get the starting point and constants.
    x, y, mu = startingPoint(Q, c, A, b, guess)
    e = np.ones_like(mu)
    sigma = .1
    tau = .95
    i = 0
    nu = 1 + tol

    #Counter check if guess sizes are wrong from startingPoint
    assert x.shape[0] == n, "A and x not aligned"
    assert y.shape[0] == mu.shape[0] == m, "y and mu not aligned"

    #Start of iteration
    while i < niter and nu >= tol:
        i += 1
        
        #Find search Direction.
        DF[-m:,n:-m] = np.diag(mu)
        DF[-m:,-m:] = np.diag(y)
        
        nu = np.dot(y, mu) / float(m)
        nu_vec = np.hstack((np.zeros(n+m), e*nu*sigma))
        lu_piv = la.lu_factor(DF)
        direct = la.lu_solve(lu_piv, nu_vec - F(x,y,mu))

        # Find step length (deltas)
        dx, dy, dmu = direct[:n], direct[n:-m], direct[-m:]
        
        # Beta, delta, alpha
        mask = dmu < 0
        beta = min(1, tau*min(1, (-mu[mask]/dmu[mask]).min())) if np.any(mask) else tau

        mask = dy < 0
        delta = min(1, tau*min(1, (-y[mask]/dy[mask]).min())) if np.any(mask) else tau

        alpha = min(beta, delta)

        # Next iteration.
        x += alpha*dx
        y += alpha*dy
        mu += alpha*dmu

        if verbose:
            print("Iteration {:0>2} nu = {}".format(i, nu))
    if i < niter and verbose:
        print("Converged in {} iterations".format(i))
    elif verbose:
        print("Maximum iterations reached")
    return x, .5*x.dot(Q).dot(x) + c.dot(x)

#Q = np.array([[1,-1],[-1,2]])
#c = np.array([-2,-6])
#A = np.array([[-1,-1],
#              [1,-2],
#              [-2,-1],
#              [1,0],
#              [0,1]])
#b = np.array([-2,-2,-3,0,0])
#guess = np.array([.5,.5])
#guess = (guess,np.ones(b.shape),np.zeros(b.shape))
#
#qInteriorPoint(Q,c,A,b,guess)
    
#%%
def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()

#%%
# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Display the resulting figure.
    """
    # Initialize the objective and constraint arrays.
    A = np.eye(n**2)
    H = laplacian(n)
    c = -np.ones(n**2) / float((n-1)**2)
    
    # Tent pole
    pole = np.zeros((n,n))
    pole[n//2-1:n//2+1,n//2-1:n//2+1] = .5
    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m, m)
    pole[mask1, mask2] = .3
    pole = pole.ravel()

    #Initial guess
    x = np.ones((n,n)).ravel()
    x = np.random.random(n**2)
    y = np.ones(n**2)
    mu = np.ones(n**2)

    # Using problem 1 function
    z = qInteriorPoint(H, c, A, pole, (x,y,mu))[0].reshape((n,n))
    
    #Plot the circus tent
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z,  rstride=1, cstride=1, color='r')
    plt.show()
    
#%%
# Problem 4
def portfolio(filename="portfolio.txt"):
    """Markowitz Portfolio Optimization

    Parameters:
        filename (str): The name of the portfolio data file.

    Returns:
        (ndarray) The optimal portfolio with short selling.
        (ndarray) The optimal portfolio without short selling.
    """
    # Markowitz portfolio optimization
    port = np.loadtxt('portfolio.txt')[:,1:]
    m,n = port.shape
    mu = 1.13
    # Returns
    R = port.mean(axis=0)
    # Covariance matrix
    Q = np.cov(port.T)
    
    # Matrix setup
    P = matrix(Q)
    q = matrix(np.zeros(n))
    b = matrix(np.array([1., mu]))
    A = np.ones((2,n))
    A[1,:] = R
    A = matrix(A)
    
    #The optimal portfolio with short selling
    sol1 = solvers.qp(P, q, A=A, b=b)
    x1 = np.array(sol1['x']).flatten()
    
    # Without short selling.
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    sol2 = solvers.qp(P, q, G, h, A, b)
    x2 = np.array(sol2['x']).flatten()
    
    return x1, x2