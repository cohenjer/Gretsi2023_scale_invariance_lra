import tensorly as tl
from tensorly.decomposition import non_negative_parafac_hals
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from tensorly import tenalg
from tlviz.factor_tools import factor_match_score as fms
from init_sparse import *
from utils import sparsify, plot_results



# hyperparams and others
verbose=False
spfacs1 = []
spfacs2 = []
spfacs3 = []
spfacs4 = []
spfacs5 = []
spfacst = []
lossf1 = []
lossf2 = []
lossf3 = []
lossf4 = []
lossf5 = []
fms1 = []
fms2 = []
fms3 = []
fms4 = []
fms5 = []

#rng = tl.check_random_state(hash("hi")%(2**32))
rng = tl.check_random_state(np.random.randint(120))
dim = 45 # 45
rank = 4 # 4
noise = 0.01
n_iter = 30
epsilon = 1e-16 # if >0, not scale invariant !!
facs =  [tl.tensor(tl.abs(rng.randn(dim,rank))) for i in range(3)]
#facs = [tl.clip(fac, a_min=2*np.median(fac))-2*np.median(fac) for fac in facs] 
# Sparse first factor
#facs[0] = tl.clip(facs[0], a_min=2*np.median(facs[0]))-2*np.median(facs[0]) # TODO use sparsify
#facs[0] = sparsify(facs[0],0.3) 
#facs[1] = sparsify(facs[1],0.3) 
#facs[2] = sparsify(facs[2],0.3) 
cp_true = tl.cp_tensor.CPTensor((None,facs)) # CP tensor
cp_true.normalize()
data = cp_true.to_tensor() + noise*tl.tensor(rng.randn(dim,dim,dim))


# rank e
gap = 2
rank_e = rank+gap
# start from same init
scale_init=True # scaling and balancing
init_cp = tl.random.random_cp(3*[dim],rank_e,random_state=rng)
init_cp.normalize()
# testing
# TODO: make proper xp
#init_cp[1][0] *=100
#init_cp[1][1] *=100
#init_cp[1][2] *=100

# decomposition
splist = [0,1e-5,1e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1,1e1,1e2,1e4]
#splist = [1e-0]
for sp in splist:
    # XP2
    spvec = [0,0,0]
    rvec = [sp/2,sp/2,sp/2]

    # XP1
    #spvec = [sp,0,0]
    #rvec = [sp/2,sp/2]
    # Scale init
    if scale_init:
        init_scaled, scale = scale_factors_fro(init_cp,data,spvec,rvec)
        print(f"Optimal init scaling is {scale}")
        # balance 2d because it does not change scale, but converse is not true
        init = balance_factors(init_scaled,spvec,rvec)
    else:
        init = init_cp

    # Compute lambda max ratio heuristic
    heuristic_regmax(init,data,spvec)

    # rescaling coeffs wrt dimensions
    #for mode in range(3):
    #    spvec[mode] *= (1/tl.shape(data)[mode]/rank)
    #    rvec[mode] *= (1/tl.shape(data)[mode]/rank)
    #
    # Balanced method 
    out1 = non_negative_parafac_hals(data, rank=rank_e, verbose=verbose, return_errors=True, sparsity_coefficients=spvec, epsilon=epsilon, n_iter_max=n_iter, init=copy(init), ridge_coefficients=rvec)
    # No reg
    out2 = non_negative_parafac_hals(data, rank=rank_e, verbose=verbose, return_errors=True, epsilon=epsilon, n_iter_max=n_iter, init=copy(init))
    # Degenerate (no l2)
    out3 = non_negative_parafac_hals(data, rank=rank_e, verbose=verbose, return_errors=True, sparsity_coefficients=spvec, epsilon=epsilon, n_iter_max=n_iter, rescale=False, init=copy(init), pop_l2=False, ridge_coefficients=rvec)
    # No balance, scaled balanced init
    out4 = non_negative_parafac_hals(data, rank=rank_e, verbose=verbose, return_errors=True, sparsity_coefficients=spvec, epsilon=epsilon, n_iter_max=n_iter, rescale=False, init=copy(init), pop_l2=True)
    # No balance regular init
    out5 = non_negative_parafac_hals(data, rank=rank_e, verbose=verbose, return_errors=True, sparsity_coefficients=spvec, epsilon=epsilon, n_iter_max=n_iter, rescale=False, init=copy(init_cp), ridge_coefficients=rvec)

    # printing and storing final errors
    # TODO: careful, loss normalized by tensor norm
    print(f"loss for sparse ridge scaled: {out1[1][-1]}")
    print(f"loss for no sparsity : {out2[1][-1]}")
    print(f"loss for sparse ridge no scaled ++init: {out3[1][-1]}")
    print(f"loss for no ridge : {out4[1][-1]}")
    print(f"loss for sparse ridge no scaled : {out5[1][-1]}")
    lossf1.append(out1[1][-1])
    lossf2.append(out2[1][-1])
    lossf3.append(out3[1][-1])
    lossf4.append(out4[1][-1])
    lossf5.append(out5[1][-1])

    # Plotting error curves
    #plt.figure()
    #plt.subplot(2,2,1)
    #plt.plot(out1[1])
    #plt.subplot(2,2,2)
    #plt.plot(out2[1])
    #plt.subplot(2,2,3)
    #plt.plot(out3[1])
    #plt.subplot(2,2,4)
    #plt.plot(out4[1])

    cp1 = out1[0]
    cp2 = out2[0]
    cp3 = out3[0]
    cp4 = out4[0]
    cp5 = out5[0]
    #plot_results(cp1,cp2,cp3,facs)

    # normalization
    #cp1.normalize()
    #cp2.normalize()
    #cp3.normalize()
    #plot_results(cp1,cp2,cp3,facs)

    # plot sparsity level (first factor)
    tol = 2*epsilon
    spfacs1.append(tl.sum(cp1[1][0]>tol))
    spfacs2.append(tl.sum(cp2[1][0]>tol))
    spfacs3.append(tl.sum(cp3[1][0]>tol))
    spfacs4.append(tl.sum(cp4[1][0]>tol))
    spfacs5.append(tl.sum(cp5[1][0]>tol))
    spfacst.append(tl.sum(facs[0]>0))
    print(f"Sparsity levels: (sp value: {sp}), rescaled {spfacs1[-1]}, unreg {spfacs2[-1]}, no rescale init++ {spfacs3[-1]}, l1 degen {spfacs4[-1]}, no rescale {spfacs4[-1]}")

    # Evaluate factor match score
    fms1.append(fms(cp1,cp_true))
    fms2.append(fms(cp2,cp_true))
    fms3.append(fms(cp3,cp_true))
    fms4.append(fms(cp4,cp_true))
    fms5.append(fms(cp5,cp_true))

offset = splist[1]
splist = [elem + offset for elem in splist]
plt.figure()
plt.subplot(131)
plt.title("sparsity of first factor")
plt.semilogx(splist, spfacs1)
plt.semilogx(splist, spfacs2)
plt.semilogx(splist, spfacs3)
plt.semilogx(splist, spfacs4)
plt.semilogx(splist, spfacs5)
plt.semilogx(splist, spfacst)
plt.xlabel('sparsity +1')
plt.legend(['ridge scale','no ridge', 'ridge no-scale init++', 'sparse(*) no ridge', 'ridge no-scale', 'true sparsity level'])

plt.subplot(132)
plt.title("Final loss values")
plt.semilogx(splist, lossf1)
plt.semilogx(splist, lossf2)
plt.semilogx(splist, lossf3)
plt.semilogx(splist, lossf4)
plt.semilogx(splist, lossf5)
plt.xlabel(f'sparsity + {offset}')
plt.legend(['ridge scale','no ridge', 'ridge no-scale init++', 'sparse(*) no ridge', 'ridge no-scale'])

plt.subplot(133)
plt.title("Final fms values")
plt.semilogx(splist, fms1)
plt.semilogx(splist, fms2)
plt.semilogx(splist, fms3)
plt.semilogx(splist, fms4)
plt.semilogx(splist, fms5)
plt.xlabel(f'sparsity + {offset}')
plt.legend(['ridge scale','no ridge', 'ridge no-scale init++', 'sparse(*) no ridge', 'ridge no-scale'])
# Bazooka
plt.show()
