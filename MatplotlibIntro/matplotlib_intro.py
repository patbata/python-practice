# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
Patricia D. Bata
BUDS Program 2019
July 27, 2019
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

#%%
# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    return np.random.normal(size=(n,n)).mean(axis=1).var()

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    
    Returns:
        plot shown with domain = [100,1000] intervals of 100 and var_of_means
        function.
    """
    # Set the domain of the function as [100,1000] intervals of 100
    n = np.arange(100,1100,100)
    # Plots the domain set with the var_of_means for each point in domain.
    plt.plot(n,[var_of_means(i) for i in n])
    return  plt.show()
    
#%%
# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    
    Returns:
        a single plot shown where the three functions are:
            sin(x) is blue, cos(x) is orange, arctan(x) is green
    """
    #Set domain as (-pi,pi) with 100 data points including limits
    x = np.linspace(-2*np.pi,2*np.pi,100)
    # Plot sin(x) vs x
    plt.plot(x,np.sin(x))
    # Plot cos(x) vs x
    plt.plot(x,np.cos(x))
    # Plot arctan(x) vs x
    plt.plot(x,np.arctan(x))
    return plt.show()

#%%
# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
           
        Returns:
            plot showing function 1/x-1 with a break at x = 1.
    """
    # First domain set to [-2,1) not including 1 (endpoint=False) with 500 pts
    x1 = np.linspace(-2,1,500,endpoint=False)
    # Set an intermediate array set to [1,6] with 500 pts (includes 1)
    x = np.linspace(1,6,500)
    # Second domain, removes 1 from intermediate array x
    x2 = x[1:]
    # Sets f1 as the output of first domain and f2 for second domain
    f1,f2 = ([(1/(i-1)) for i in x1],[(1/(i-1)) for i in x2])
    # Plots f1 vs x1 with magenta line and linewidth = 4 
    plt.plot(x1,f1,'m--',linewidth=4)
    # Plots f2 vs x2 with magenta line and linewidth = 4
    plt.plot(x2,f2,'m--',linewidth=4)
    # Set upper and lower limits of x-axis as -2 and 6 respectively
    plt.xlim(-2,6)
    # Set upper and lower limits of y-axis as -6 and 6 respectively
    plt.ylim(-6,6)
    return plt.show()

#%%
# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    
    Returns:
        A single figure with 4 plots: sin(x), sin(2x), 2sin(x), 2sin(2x).
    """
    # Sets domain from [0,2pi] with 500 points
    x = np.linspace(0,2*np.pi,500)
    # Sets axes for a 2x2 plot.
    fig, axes = plt.subplots(2,2)
    #Plot all the functions
    axes[0,0].plot(x, np.sin(x),'g')
    axes[0,1].plot(x, np.sin(2*x),'r--')
    axes[1,0].plot(x, 2*np.sin(x),'b--')
    axes[1,1].plot(x, 2*np.sin(2*x),'m:')
    #Set the titles of the subplots
    axes[0,0].set_title("sin(x)")
    axes[0,1].set_title("sin(2x)")
    axes[1,0].set_title("2sin(x)")
    axes[1,1].set_title("2sin(2x)")
    #Set the axis limits for all subplots
    for i in [0,1]:
        for j in [0,1]: axes[i,j].axis([0,2*np.pi,-2,2])
    #Figure Title
    fig.suptitle("Problem 4: Sine Functions")
    plt.tight_layout()
    return plt.show()

#%%
# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    #Loads FARS.npy stores in 'fars' variable
    fars=np.load("FARS.npy")
    fig, axes = plt.subplots(1,2)
    #Scatter of longitude vs. latitude
    axes[0].plot(fars[:,1],fars[:,2], 'k,')
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_aspect("equal")
    
    #Histogram of hours of day
    axes[1].hist(fars[:,0],bins=np.arange(0,25)-0.5)
    axes[1].axis([-1,24,0,9000])
    axes[1].set_xlabel("Hour of the Day (Military Time)")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    return plt.show()

#%%
# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    # Sets domain as [-2pi,2pi] with 200 points in total
    x=np.linspace(-2*np.pi,2*np.pi,200)
    # Stores a copy of x in y
    y=x.copy()
    # Creates a meshgrid of X and Y
    X,Y = np.meshgrid(x,y)
    # Function of sin(x)sin(y)/xy using the meshgrid X and Y
    Z = (np.sin(X)*np.sin(Y))/(X*Y)
    
    # Set the figure to have two subplots with format 1 row, 2 columns
    # Heat Map at 1,1
    plt.subplot(121)
    # Heat map plotting with color set to "gnuplot2"
    plt.pcolormesh(X,Y,Z,cmap="gnuplot2")
    # Puts in a colorbar()
    plt.colorbar()
    # Set upper and lower limits of x as -2pi and 2pi
    plt.xlim(-2*np.pi,2*np.pi)
    # Set upper and lower limits of y as -2pi and 2pi
    plt.ylim(-2*np.pi,2*np.pi)
    
    #Contour Plot at 1,2
    plt.subplot(122)
    # Creates the contour plot at level 20 with color set to "spectral"
    plt.contour(X,Y,Z,20,cmap="Spectral")
    # Inserts a colorbar()
    plt.colorbar()
    # Set upper and lower limits of x as -2pi and 2pi
    plt.xlim(-2*np.pi,2*np.pi)
    # Set upper and lower limits of y as -2pi and 2pi
    plt.ylim(-2*np.pi,2*np.pi)
    return plt.show()
    