# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
from imageio import imread
#from matplotlib import pyplot as plt

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    #Set elements as float to not equal 0 in round off
    A = A.astype("float64")
    #Get shape of matrix
    m, n = A.shape
    #Get eigenvalues and vectors using la.eig
    lamb, V = la.eig(A.conj().T@A)
    sigma = np.sqrt(lamb)
    #Sort sigma and V columns by greatest to least of sigma
    sigma, V = sigma[np.argsort(-sigma)], V[:,np.argsort(-sigma)]
    #Find sigma greater than set tolerance
    r = (sigma > tol).sum()
    #Cut until sigma > tol and do to V and U
    sigma1 = sigma[:r]
    V1 = V[:,:r]
    U1 = A@V1/sigma1
    return U1, sigma1, V1.conj().T
    
#%%
# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    #Set elements of A as float
    A = A.astype("float64")
    #Set the x, y, theta, r of circle
    theta = np.linspace(0,2*np.pi, 200)
    x = np.cos(theta)
    y = np.sin(theta)
    S = np.array([x,y])
    #Set E given in problem
    E = np.array([[1,0,0],[0,0,1]])
    #Get U, s, Vh of A using la.svd()
    U, s, Vh = la.svd(A)
    #Set titles of subplots in list
    titles = ["S", "$\mathregular{V^H S}$", "$\mathregular{\Sigma V^H S}$", "$\mathregular{U \Sigma V^H S}$"]
    #Plot the transformations
    fig, axes = plt.subplots(2,2)
    axes[0,0].plot(S[0], S[1], E[0], E[1])
    S, E = Vh@S, Vh@E
    axes[0,1].plot(S[0], S[1], E[0], E[1])
    S, E = np.diag(s)@S, np.diag(s)@E
    axes[1,0].plot(S[0], S[1], E[0], E[1])
    S, E = U@S, U@E
    axes[1,1].plot(S[0], S[1], E[0], E[1])
    #Set ylimit
    axes[1,0].set_ylim(-4,4)
    #set aspect as equal
    [axes[int(np.floor(i/2)), i%2].set_aspect("equal") for i in range(0,4)]
    #Get titles of subplots from list
    [axes[int(np.floor(i/2)), i%2].set_title(titles[i]) for i in range(0,4)]
    plt.suptitle("Visualizing SVD")
    plt.tight_layout()
    plt.subplots_adjust(wspace = -0.5)
    plt.show()


#%%
# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    #Convert all elements in A as float
    A = A.astype("float64")
    #Get shape of matrix (m, n)
    m, n = A.shape
    #Get SVD of A
    U, sigma, Vh = la.svd(A, full_matrices=False)
    #Get all rank of matrix that's less than 0
    r = (sigma > 0).sum()
    #RAISE AN ERROR if the set approximation is larger than the rank of the sigma matrix
    if s > r:
        raise ValueError("s > rank(A)")
    #Sort based on sigma to make sure
    sigma, Vh, U = sigma[np.argsort(-sigma)], Vh[:,np.argsort(-sigma)], U[:,np.argsort(-sigma)]
    #Cut til the sth element/row
    s1, V1, U1 = sigma[:s], Vh[:s,:], U[:,:s]
    #Get best approximated rank
    As = U1 @ np.diag(s1) @ V1
    entries = (m*s + s + n*s)
    return As, entries

#%%
# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    #Set elements of A as float
    A = A.astype("float64")
    #Get shape of matrix A
    m, n = A.shape
    #Get SVD of matrix A
    U, sigma, Vh = la.svd(A, full_matrices=False)
    #Sort sigma, Vh, U based on greatest to least of sigma
    sigma, Vh, U = sigma[np.argsort(-sigma)], Vh[:,np.argsort(-sigma)], U[:,np.argsort(-sigma)]
    #Raise an error if the error set is greater than the smallest value in sigma
    if err < sigma[len(sigma)-1]:
        raise ValueError("A cannot be approximated within the tolerance by a matrix of lesser rank.")
    #Get the argument where sigma is greater than the error
    s = np.argmax(sigma < err)
    #Cut S, V, U until the sth row/element 
    s1, V1, U1 = sigma[:s], Vh[:s,:], U[:,:s]
    #Get approximated matrixx
    As = U1 @ np.diag(s1) @ V1
    #Get number of entries
    entries = (m*s + s + n*s)
    return As, entries

#%%
# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    #Read image name
    image = imread(filename) / 255
    #Set subplots
    fig, axes = plt.subplots(1,2)
    #For case when image is black and white
    if len(image.shape) == 2:
        #get best approximated matrix
        As, e = svd_approx(image, s)
        #plot the two images with titles
        axes[0].imshow(image, cmap = "gray")
        axes[0].set_title("Original Image", y=-0.1)
        axes[1].imshow(As, cmap = "gray")
        axes[1].set_title("Rank {} Approximation".format(s), y=-0.1)
        [axes[i].axis("off") for i in [0,1]]
    #For case when image is colored
    elif len(image.shape) == 3:
        #Stack all of the layers that were approximated
        cimage = np.dstack([svd_approx(image[:,:,i], s)[0] for i in range(0,3)])
        #Set all the values in cimage that are less than 0 to 0 and greater than 1 to 1
        cimage[cimage < 0] = 0
        cimage[cimage > 1] = 1
        #Get the number of elements in approximated matrix
        e = sum([svd_approx(image[:,:,i], s)[1] for i in range(0,3)])
        #Plot properties/labels
        axes[0].imshow(image)
        axes[0].set_title("Original Image", y=-0.1)
        axes[1].imshow(cimage)
        axes[1].set_title("Rank {} Approximation".format(s), y=-0.1)
        [axes[i].axis("off") for i in [0,1]]
    #Set plot
    plt.suptitle("Difference in Number of Entries is {}".format(image.size - e),y=0.88)
    plt.tight_layout()
    plt.show()