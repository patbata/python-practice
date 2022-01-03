# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
Patricia D. Bata
BUDS Program 2019
September 6, 2019
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath as cmath

# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    #Set elements of A as float to avoid roundoff errors
    A = A.astype("float64")
    #Get the QR decomposition of A using la.qr
    Q, R = la.qr(A, mode = 'economic')
    #Use solve_triangular to get least squares
    return la.solve_triangular(R, Q.T@b)


# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    #Load the housing data into "house"
    house = np.load("housing.npy")
    #Column stack the x of data set with ones columns (constants)
    A = np.column_stack((house[:, 0], np.ones(len(house))))
    #set b as the y of the housing data
    b = house[:,1]
    #use least_squares function to get slope and constant in y= mx+b1
    m, b1 = least_squares(A, b)
    #Create scatter plot of the "house" data 
    plt.scatter(house[:,0],b)
    #plot the line fit
    x = np.linspace(0,16,33)
    plt.plot(x,m*x+b1, color = 'r')
    plt.xlabel("Year")
    plt.ylabel("Price Index")
    plt.title("Purchase-only Housing Price Index for U.S. from 2000 to 2010")

# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    house = np.load("housing.npy")
    #Set the x and b
    x = np.linspace(0,16,33)  
    b = house[:, 1]
    #Polynomial degrees to be used
    poly = [3, 6, 9, 12]
    fig, ax = plt.subplots(nrows = 2, ncols=2) 
    #plot all of the 4 polynomial fits
    for i in range(0,len(poly)):
        A = np.vander(house[:,0], poly[i]+1)
        f = np.poly1d(la.lstsq(A,b)[0])
        g = np.poly1d(np.polyfit(house[:,0], house[:,1], deg=poly[i]))
        ax[int(np.floor(i/2)), i%2].plot(house[:,0], b, "xb")
        ax[int(np.floor(i/2)), i%2].plot(x, f(x), '+r')
        ax[int(np.floor(i/2)), i%2].plot(x, g(x), color = 'g')
        ax[int(np.floor(i/2)), i%2].set_xlabel("Year")
        ax[int(np.floor(i/2)), i%2].set_ylabel("Price Index")
        ax[int(np.floor(i/2)), i%2].set_title("Polynomial Fit with Degree = " + str(poly[i]))
    #SEt plot and figure titles/legends
    fig.legend(["Original","Vander", "Polyfit" ], loc = "center left", bbox_to_anchor = (1,0.5))
    plt.suptitle("Polynomial Fit of Purchase-only Housing Price", y=1.05)
    plt.tight_layout()

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    #load the ellipse points into array
    xk, yk = np.load("ellipse.npy").T
    #Columns stack in order to get a, b, c, d, e for eqn of ellipse 
    A = np.column_stack((xk**2, xk, xk*yk, yk, yk**2))
    #SInce ellipse eqn =1, set b as all 1s
    b = np.ones_like((xk))
    #Calculate the least squares solution and solve for a, b, c, d, e
    a, b, c, d, e = la.lstsq(A, b)[0]
    #Plot the best fit ellipse and the data points
    plot_ellipse(a, b, c, d, e)
    plt.plot(xk, yk, 'k*')      # Plot the data points
    plt.show()

# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    #Get the shape of A
    m, n = A.shape
    #Get random guess 
    x0 = np.random.random((n,1))
    #normalize the initial guess vector
    x0 = x0/la.norm(x0)
    #Power method algorithm
    for k in range(0, N):
        x0 = np.column_stack((x0, A@x0[:, k]))
        x0[:,k+1] /= la.norm(x0[:,k+1])
        if la.norm(x0[:,k+1] - x0[:,k]) < tol:
            break
    return x0[:,k+1].T@A@x0[:, k+1], x0[:,k+1]

# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    #get shape of matrix A
    m, n = A.shape
    #Get the hessenberg of A into S
    S = la.hessenberg(A)
    #Start of qr_algorithm
    for k in range(0, N):
        #Get QR decomposition of S
        Q, R = la.qr(S)
        S = R@Q
    #Set eigenvalues list
    eigs = []
    #Set counter
    i = 0
    while i < n:
        if S[i,i] == np.diag(S)[len(np.diag(S))-1]:     #Checks if S[i,i] is last in diagonal
            eigs.append(S[i,i])
        #When 1x1 based on definition in problem
        elif np.abs(S[i+1,i]) < tol:
            eigs.append(S[i,i])
        #Get the 2x2 determinant/eigenvalues
        else:
            #a,b,c,d as elements of 2x2 matrix; stored values
            a, b, c, d = S[i, i], S[i, i+1], S[i+1, i], S[i+1, i+1]
            #Compute the b and c of the quadratic formula for evalues
            b, c = -(d+a), (a*d - b*c)
            #Append the +- values from quadratic in the eigs lis
            eigs.append((-b + cmath.sqrt(b**2 - (4*c)))/2)
            eigs.append((-b - cmath.sqrt(b**2 - (4*c)))/2)
            #Set counter +1
            i += 1
        #Set counter +1
        i+=1
    return eigs