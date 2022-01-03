# qr_decomposition.py
"""Volume 1: The QR Decomposition.
Patricia D. Bata
BUDS Program 2019
September 6, 2019
"""
import numpy as np
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    #Convert all elements of A to float to avoid roundoff error
    A = A.astype("float64")
    #Get n of the matrix
    n = A.shape[1]
    #Copy A into Q as placeholder
    Q = np.copy(A)
    #Get a matrix R with size n,n based on size of A
    R = np.zeros((n,n))
    for i in range(0, n):
        #Set the [i,i] value in R as the normal of each row in Q
        R[i,i] = la.norm(Q[:,i])
        #Divide each element in Q with the normal in R
        Q[:,i] = Q[:,i]/R[i,i]
        #Get final Q and R
        for j in range(i+1, n):
            R[i,j] = (Q[:,j]).T @ Q[:,i]
            Q[:,j] = Q[:,j] - (R[i,j] * Q[:,i])
    return Q, R    


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    #Get the absolute determinant of the A using QR decomposition of A
    return np.prod(np.diag(qr_gram_schmidt(A)[1]))
  
    
# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    #Set all elements of A as float
    A = A.astype("float64")
    #Get Q and R using qr_gram_schmidt(A)
    Q, R = qr_gram_schmidt(A)
    #set x and R where x = y and diagonals of R are set to 1 for easy computation 
    x, R = Q.T @ b/np.diag(R), (R.T/np.diag(R)).T
    #Vectorized back substitution
    for i in range(1,len(x)+1):
        x[len(x)-i] -= R[len(x)-i, (len(x)-i+1):] @ x[(len(x)-i+1):]
    return x


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    #Set all elements of A as float to avoid round off error
    A = A.astype("float64")
    #Define function sign to get sign of x
    sign = lambda x: 1 if x >= 0 else -1
    #Copy A into R
    R = np.copy(A)
    #Get an identity matrix in the shape of A's rows into Q
    Q = np.eye(A.shape[0])
    #Householder algorithm
    for k in range(0, A.shape[1]):
        u = np.copy(R[k:,k])
        u[0] += sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        R[k:,k:] = R[k:,k:] - 2*np.outer(u, u.T @ R[k:,k:])
        Q[k:,:] = Q[k:,:] - 2*np.outer(u, u.T @ Q[k:,:])
    return Q.T, R

# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """ 
    #Set the elements of A as a float
    A = A.astype("float64")    
    #Define function to get the sign of x
    sign = lambda x: 1 if x >= 0 else -1
    #Copy the matrix A into variable H
    H = np.copy(A)
    #Get an identity matrix with same shape as A's rows into Q
    Q = np.eye(A.shape[0])
    #Hessenberg algorithm
    for k in range(0, A.shape[1]-2):
        u = np.copy(H[k+1:,k])
        u[0] += sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u, u.T @ H[k+1:, k:])
        H[:,k+1:] = H[:, k+1:] - 2*np.outer(H[:,k+1:]@u, u.T)
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u, u.T @ Q[k+1:, :])
    return H, Q.T