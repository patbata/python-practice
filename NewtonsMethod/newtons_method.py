# newtons_method.py
"""Volume 1: Newton's Method.
Patricia D. Bata
BUDS Program 2019
September 9, 2019
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    #Initial set x1 as a value x0
    x1 = x0
    #Check if initial guess is scalar; 1D
    if np.isscalar(x0):
        for k in range(maxiter):
            #Reassign x1 as x0 to do Newton's
            x0 = x1
            #Newton's method
            x1 = x0 - alpha*(f(x0)/Df(x0))
            #If difference between guesses is less than tol, accept new x1
            if abs(x1 - x0) < tol:
                break
        return x0, abs(x1 - x0) < tol, k+1
    #Only other option is a scalar; check norm difference
    else:
        for k in range(maxiter):
            #Reassign x1 as x0 to do Newton's
            x0 = x1
            #Use la.solve to solve Df*x = f(x)
            x1 = x0 - alpha*la.solve(Df(x0), f(x0))
            #If normal difference is less than tol, accept
            if la.norm(x1 - x0) < tol:
                break
        return x0, la.norm(x1 - x0) < tol, k+1


# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    #Set initial guess
    r_0 = 0.1
    #Defined function and derivative from problem
    f = lambda r: P1*((1+r)**N1 - 1) - P2*(1 - (1+r)**(-N2))
    Df = lambda r: P1*N1*(1+r)**(N1-1) - P2*N2*(1+r)**(-N2-1)
    return newton(f, r_0, Df)


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    #Set list which contains number of iterations for a given alpha
    iters = []
    #Alpha range of numbers
    rangea = np.linspace(0.01,1,1000)
    for alpha in rangea:
        #Renamed x0 to xo so that x0 does not get replaced
        xo = x0
        x1 = xo
        #Iterate Newton
        for k in range(maxiter):
            xo = x1
            x1 = xo - alpha*(f(xo)/Df(xo))
            if abs(x1 - xo) < tol:
                break
        iters.append(k+1)
    #Plot of iterations and alphas
    plt.plot(rangea, iters)
    plt.xlabel("alpha")
    plt.ylabel("Number of Iterations")
    return rangea[np.argmin(iters)]

# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    #Defined function and derivative
    f = lambda x: np.array([4*x[0]*x[1] - x[0], -x[0]*x[1]+1-x[1]**2])
    Df = lambda x: np.array([[4*x[1]-1, 4*x[0]], [-x[1], -x[0]-2*x[1]]])
    #Set domain and range
    X = np.linspace(-0.4, -0.0001, 32)
    Y = np.linspace(0.45, 0.001, 32)
    #Set domain
    tol = 1e-5
    #For loop for checking where optimal initial guess is
    for y in Y:
        #Check all x's while keeping y constant
        x0 = np.stack((X, y*(np.ones_like(X))))
        #Find the newton values for each x, y combo; alpha = 1
        n = np.column_stack([newton(f, x0[:,i], Df, alpha = 1)[0] for i in range(0, X.shape[0])])
        #Find root [0,1]
        c1 = (abs(n - np.array([[0],[1]])) < tol).all(axis = 0)
        #Find root [0,-1]
        c2 = (abs(n - np.array([[0],[-1]])) < tol).all(axis = 0)
        #Find the newton values for each x, y combo; alpha = 0.55
        n1 = np.column_stack([newton(f, x0[:,i], Df, alpha = 0.55, maxiter = 40)[0] for i in range(0, len(X))])
        #Find root [3.75, 0.25]
        c3 = (abs(n1 - np.array([[3.75],[0.25]])) < tol).all(axis = 0)
        #Satisfy all conditions c1, c2, c3
        if ((c1 | c2) & c3).any():
            break
    return x0[:,((c1 | c2) & c3).argmax()]

# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    # Real parts.
    x_real = np.linspace(-1.5, 1.5, 500) 
    # Imaginary parts
    x_imag = np.linspace(-1.5, 1.5, 500)
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    # Combine real and imaginary parts.
    X_0 = X_real + 1j*X_imag 
    #Set initial value for X_1
    X_1 = X_0
    for i in range(iters):
        #Assign new X_1 to old X_0 for Newton's
        X_0 = X_1
        X_1 = X_0 - f(X_0)/Df(X_0)
    #Get nearest root by getting argument of minimum of stacked matrices
    root = np.array([np.abs(X_1 - zeros[i]) for i in range(0, len(zeros))]).argmin(axis = 0)
    #Plot the colormesh
    fig, ax = plt.subplots(figsize = (5,5))
    plt.pcolormesh(X_real, X_imag, root, cmap = "brg")
    plt.show()