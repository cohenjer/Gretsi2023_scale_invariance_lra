import numpy as np
import matplotlib.pyplot as plt

def sparsify(M, s=0.5, epsilon=0):
    """Adds zeroes in matrix M in order to have a ratio s of nnzeroes/nnentries.

    Parameters
    ----------
    M : 2darray
        The input numpy array
    s : float, optional
        the sparsity ratio (0 for fully sparse, 1 for density of the original array), by default 0.5
    """    
    vecM = M.flatten()
    # use quantiles
    val = np.quantile(vecM, 1-s)
    # put zeros in M
    M[M<val]=epsilon
    return M

# graphics
def plot_results(cp1,cp2,cp3,facs):
    plt.figure()
    plt.subplot(4,3,1)
    plt.plot(cp1[1][0])
    plt.ylabel('scaling')
    plt.subplot(4,3,2)
    plt.plot(cp1[1][1])
    plt.subplot(4,3,3)
    plt.plot(cp1[1][2])
    plt.subplot(4,3,4)
    plt.plot(cp2[1][0])
    plt.ylabel('sparsity no ridge')
    plt.subplot(4,3,5)
    plt.plot(cp2[1][1])
    plt.subplot(4,3,6)
    plt.plot(cp2[1][2])
    plt.subplot(4,3,7)
    plt.plot(cp3[1][0])
    plt.ylabel('sparsity ridge no scaling')
    plt.subplot(4,3,8)
    plt.plot(cp3[1][1])
    plt.subplot(4,3,9)
    plt.plot(cp3[1][2])
    plt.subplot(4,3,10)
    plt.plot(facs[0])
    plt.ylabel('True facs')
    plt.subplot(4,3,11)
    plt.plot(facs[1])
    plt.subplot(4,3,12)
    plt.plot(facs[2])