# solutions.py
"""Volume 2: Gradient Descent Methods.
Patricia D. Bata
BUDS Program
11/19/2019
"""

import numpy as np
import scipy.optimize as opt
from scipy import linalg as la
from matplotlib import pyplot as plt

#%%
# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Convert x0 to float 
    x0 = np.array([float(x) for x in x0])
    #Function for optimization (next step alpha)
    f1 = lambda a: f(x0 - a * Df(x0).T)
    #Default status for convergece
    stat = False
    
    #Iterations
    for k in range(0, maxiter):
        #Minimizing alpha
        a = opt.minimize_scalar(f1).x
        x0 -= a*Df(x0).T
        #Condition for convervence
        if la.norm(Df(x0)) < tol:
            stat = True
            break
            
    return x0, stat, k


#f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
#Df = lambda x: np.array([4*x[0]**3,4*x[1]**3,4*x[2]**3])
#x0 = np.array([50.,20.,20.])
##Test easy func
#steepest_descent(f, Df, x0, tol = 1e-10)
#
#f_rosen = lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
#Df_rosen = lambda x: np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2, 200*(x[1] - x[0]**2)])
#x0 = np.array([0,0])
##Test hard func
#steepest_descent(f_rosen, Df_rosen, x0, tol = 1e-5, maxiter = 10000)

#%%
# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Convert x0 to float 
    x0 = np.array([float(x) for x in x0])
    
    #Defining initial vectors
    r0 = Q@x0 - b
    d = -r0
    n = b.size
    stat = False
    
    #Start of iteration
    for k in range(0, n+1):
        a = (r0@r0)/(d.T@Q@d)
        x0 += a*d
        rk = r0 + a*Q@d
        beta = (rk@rk)/(r0@r0)
        d = -rk + beta*d
        r0 = rk
        
        #Condition for convergence
        if la.norm(rk) < tol:
            stat = True
            break
    return x0, stat, k


#Q = np.array([[2.,0.],[0.,4.]])
#b = np.array([1.,8.])
#x0 = np.array([0.,0.])
#conjugate_gradient(Q,b,x0)
#x = conjugate_gradient(Q,b,x0, tol = 1e-5)[0]
#print(x)
#np.allclose(Q@x,b)

#%%
# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Convert x0 to float 
    x0 = np.array([float(x) for x in x0])
    
    #Define initial vectors
    r0 = -df(x0).T
    d0 = r0
    f1 = lambda a: f(x0 + a*d0)
    
    #First guess
    a = opt.minimize_scalar(f1).x
    x0 += a*d0
    stat = False
    
    #Begin iteration
    for k in range(0,maxiter):
        rk = -df(x0).T
        beta = (rk.T@rk) / (r0.T@r0)
        d0 = rk + beta*d0
        a = opt.minimize_scalar(f1).x
        xk = x0 + a*d0
        x0, r0 = xk, rk
        if  la.norm(rk, 2) < tol:
            stat = True
            break
    return x0, stat, k
        
#f = opt.rosen
#df = opt.rosen_der
#x0 = np.array([10, 10])
##opt.fmin_cg(f, x0, fprime=df)
#nonlinear_conjugate_gradient(f, df, x0, maxiter = 1000)

#%%
# Problem 4
def prob4(filename="linregression.txt",
    x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    #Setting up arays
    linreg = np.loadtxt(filename)
    y = linreg[:,0]
    A = linreg.copy()
    A[:,0] = 1
    
    #Defining Q and b
    Q = A.T@A
    b = A.T@y
    
    #Apply conjugate_grandient
    return conjugate_gradient(Q, b, x0, tol=1e-4)[0]

prob4()
    
#%%
# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def probs(self, b, x):
        return 1/(1 + np.exp(-b[0]-b[1]*x))
    
    def loglike(self, b):
        return np.sum(np.log(1 + np.exp(-b[0]-b[1]*self.x)) + (1-self.y)*(b[0]+b[1]*self.x))
    
    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        self.x, self.y = x, y
        self.b = opt.fmin_cg(self.loglike, guess) 
    

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        return self.probs(self.b, x)


#%%
# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    #Load data
    chal = np.load(filename)
    
    #Define variables and class
    temp, prob = chal[:,0], chal[:,1]
    logit = LogisticRegression1D()
    
    #Fit and predict
    logit.fit(temp, prob, guess)
    tempz = np.linspace(30,100, 500)
    probz = logit.predict(tempz)
    prob31 = logit.predict(31)
    
    #Plot of data
    plt.plot(temp, prob, '.', label = "Previous Damage")
    plt.plot(tempz, probz, color = 'orange')
    plt.plot(31, prob31,'.', label = "P(Damage) at Launch", color = 'g')
    #Setting plot properties
    plt.xlim(25,105)
    plt.xlabel("Temperature")
    plt.ylabel("O-Ring Damage")
    plt.title("Probability of O-Ring Damage")
    plt.legend(loc= 'best')
    
    return prob31
    
#prob6()