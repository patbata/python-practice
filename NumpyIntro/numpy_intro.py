# numpy_intro.py
"""Python Essentials: Intro to NumPy.
Patricia D. Bata
BUDS Program 2019
July 26, 2019
"""

import numpy as np

#%%
def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""
    # Define A
    A=np.array([[3,-1,4],[1,5,-9]])
    # Define B
    B=np.array([[2,6,-5,3],[5,-8,9,7],[9,-3,-2,-3]])
    # Returns the matrix product of A and B (@)
    return A@B

#%%
def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    # Define the array A
    A = np.array([[3,1,4],[1,5,9],[-5,3,1]])
    # Return the result of the matrix operations: (-A)**3 + 9*(A**2) - 15*A 
    return (-A@-A@-A) + 9*(A@A) -15*A

#%%
def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    #Define array A as a 7x7 matrix with the upper triangle=1, the rest=0
    A = np.triu(np.ones((7,7)))
    #Define array B. Create a 7x7 matrix full of 5 MINUS 7x7 matrix with the
    #lower triangle values equal to 6
    B = np.full((7,7),5)-np.tril(np.full((7,7),6))
    #Return AxBxA with each element as type = np.int64
    return (A@B@A).astype(np.int64)

#%%
def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    # Makes a copy of the inputted array
    acop = np.copy(A)
    # Change all the values in A(copy) that are negative = 0
    # (set mask as acop<0)
    acop[acop<0] = 0
    # Return the copy of A with the negative numbers converted to 0
    return acop

#%%
def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    
    Returns an 8x8 matrix as per instructions above.
    """
    #Define the array A (vector of 1-5 converted to 3x2 and transposed)
    A = np.arange(6).reshape(3,2).T
    #Define the array B (start with 3x3 zero  add upper triangle 3x3 
    #                                                          filled with 3)
    B = np.zeros((3,3)) + np.tril(np.full((3,3),3))
    #Define the array C (3x3 array of 0 and add -2 to diagonal)
    C = np.zeros((3,3)) + np.diag([-2, -2, -2])
    #Stack c1 (8x3 array) with zeroes on top, A then B vertically
    c1 = np.vstack((np.zeros((3,3)),A,B))
    #Stack c2 (8x2 array) with A.T on top of a zeroes array vertically
    c2 = np.vstack((A.T,np.zeros((5,2))))
    #Stack c3 (8x3 array) with identity matrix (3x3) on top
    #                   of zeroes array (2x3) with C at the bottom, vertically    
    c3 = np.vstack((np.eye(3),np.zeros((2,3)),C))
    #Returns the horizontal stack of the three piece-wise columns
    return np.column_stack((c1,c2,c3))

#%%
def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    #Divide all the elements of A by the sum of rows by converting 
    #                                     the sum vector to a vertical vectory
    return A/(A.sum(axis=1).reshape((-1,1)))

#%%
def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    # Loads grid.npy as grid
    grid = np.load("grid.npy")
    #Max product of 4 adjacent numbers horizonally (left and right)
    R = np.max(grid[:,:-3]*grid[:,1:-2]*grid[:,2:-1]*grid[:,3:])
    #Max product of 4 adjacent numbers vertically (up and down)
    C = np.max(grid[:-3,:]*grid[1:-2,:]*grid[2:-1,:]*grid[3:,:])
    #Max product of 4 adjacent numbers diagonally (down to the right)
    DR = np.max(grid[:-3,:-3]*grid[1:-2,1:-2]*grid[2:-1,2:-1]*grid[3:,3:])
    #Max product of 4 adjacent numbers diagonally (down to the left)
    DL = np.max(grid[:-3,3:]*grid[1:-2,2:-1]*grid[2:-1,1:-2]*grid[3:,:-3])
    # Returs the ultimate max of the 4 max values from the sliced arrays
    return max(R, C, DR, DL)
#%%