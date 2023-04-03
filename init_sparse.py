import tensorly as tl
from tensorly.cp_tensor import unfolding_dot_khatri_rao as mttkrp
import numpy as np

def opt_balance(regs, hom_deg):
    '''
    Computes the multiplicative constants to scale factors columnwise such that regularizations are balanced.
    The problem solved is 
        min_{a_i} \sum a_i s.t.  \prod a_i^{p_i}=q
    where a_i = regs[i] and p_i = hom_deg[i]

    Parameters
    ----------
    regs: 1d np array
        the input regularization values
    hom_deg: 1d numpy array
        homogeneity degrees of each regularization term

    Returns
    -------
    scales: list of floats
        the scale to balance the factors. Its product should be one (scale invariance).
    '''
    # 0. If reg is zero, do not scale
    if tl.prod(regs)==0:
        # TODO warning
        print(f"No rescaling because regularization is null")
        return [1 for i in range(len(regs))]


    # 1. compute q
    prod_q = tl.prod(regs**(1/hom_deg))

    # 2. compute beta
    beta = (prod_q*tl.prod(hom_deg**(1/hom_deg)))**(1/tl.sum(1/hom_deg))

    # 3. compute scales
    scales = [(beta/regs[i]/hom_deg[i])**(1/hom_deg[i]) for i in range(len(regs))]

    return scales


def balance_factors(cp_tensor,sparsity_coefficients,ridge_coefficients):
    factors = np.copy(cp_tensor[1])
    n_modes = len(cp_tensor[1])
    rank = cp_tensor.rank
    hom_deg = tl.tensor([1.0*(sparsity_coefficients[i]>0) + 2.0*(ridge_coefficients[i]>0) for i in range(n_modes)]) # +1 for the core
    for q in range(rank):
        # we can safely rescale (not exact but its fine)
        regs = [sparsity_coefficients[i]*tl.sum(tl.abs(factors[i][:,q])) + ridge_coefficients[i]*tl.norm(factors[i][:,q])**2 for i in range(n_modes)]
        scales = opt_balance(tl.tensor(regs),hom_deg)
        #print(scales)
        if tl.abs(tl.prod(scales)-1)>1e-8:
            print("error in scaling")
            print(scales)
        for submode in range(n_modes):
            factors[submode][:,q] = factors[submode][:,q]*scales[submode]
    
    return tl.cp_tensor.CPTensor((None, factors))


def scale_factors_fro(cp_tensor,data,sparsity_coefficients,ridge_coefficients):
    '''
    Optimally scale [A,B,C] in 
    
    min_x \|data - x^{n_modes} [A_1,A_2,A_3]\|_F^2 + \sum_i sparsity_coefficients_i \|A_i\|_1 + \sum_j ridge_coefficients_j \|A_j\|_2^2

    This avoids a scaling problem when starting the separation algorithm, which may lead to zero-locking.
    The problem is solved by finding the positive roots of a polynomial.
    '''
    factors = np.copy(cp_tensor[1])
    n_modes = len(cp_tensor[1])
    l1regs = [sparsity_coefficients[i]*tl.sum(tl.abs(factors[i])) for i in range(n_modes)]
    l2regs= [ridge_coefficients[i]*tl.norm(factors[i])**2 for i in range(n_modes)]
    # We define a polynomial
    # a x^{2n_modes} + b x^{n_modes} + c x^{2} + d x^{1}
    # and find the roots of its derivative, compute the value at each one, and return the optimal scale x and scaled factors.
    a = cp_tensor.norm()**2
    b = -2*tl.sum(data*cp_tensor.to_tensor())
    c = sum(l2regs)
    d = sum(l1regs)
    poly = [0 for i in range(2*n_modes+1)]
    poly[1] = d
    poly[2] = c
    poly[n_modes] = b
    poly[2*n_modes] = a
    poly.reverse()
    grad_poly = [0 for i in range(2*n_modes)]
    grad_poly[0] = d
    grad_poly[1] = 2*c
    grad_poly[n_modes-1] = n_modes*b
    grad_poly[2*n_modes-1] = 2*n_modes*a
    grad_poly.reverse()
    roots = np.roots(grad_poly)
    current_best = np.Inf
    best_x = 0
    for sol in roots:
        if sol.imag<1e-16:
            sol = sol.real
            if sol>0:
                val = np.polyval(poly,sol)
                if val<current_best:
                    current_best = val
                    best_x = sol
    if current_best==np.Inf:
        print("No solution to scaling !!!")
        return cp_tensor, None

    # We have the optimal scale
    for i in range(n_modes):
        factors[i] *= best_x

    return tl.cp_tensor.CPTensor((None, factors)), best_x


def heuristic_regmax(cp_tensor, data, l1regs):
    regmax = np.Inf    
    for mode in range(len(cp_tensor[1])):
        if l1regs[mode]:
            regmax = min(regmax, tl.max(tl.abs((mttkrp(data, cp_tensor, mode)))))
    print(f"The current regularization strenght is about {max(l1regs)/regmax*100}%")
    return max(l1regs)/regmax, regmax
    
