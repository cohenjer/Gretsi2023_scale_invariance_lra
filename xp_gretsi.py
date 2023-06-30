import tensorly as tl
from tensorly.decomposition import non_negative_parafac_hals #check out the correct version
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from tensorly import tenalg
from tlviz.factor_tools import factor_match_score as fms
from init_sparse import *
from utils import sparsify, plot_results

from shootout.methods.runners import run_and_track


# hyperparams and others
verbose=False
algorithms = ['ridge balance', 'unregularized', 'ridge no-balance init++', 'sparse(*) no ridge', 'ridge no-balance']
name = "./results/xp_29-06-2023"

# XP1,2: Comparison of behavior for sparse and low-rank inducing penalties, balance vs no balance (init++ or not) vs degenerate vs vanilla
variables = dict({
    #"sp": [0,1e-5,1e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1,1e1,1e2,1e4], old xp
    "sp": [0,1e-2,1e-1,1e-0,5e-0,1e1,5e1,1e2,1e3,1e4,1e5,1e6],
    "xp": ["sparse", "lowrank"],
    "scale_init": True,
    "rank": 4,
    "e_rank": 6,
    "dim": 30,
    "noise": 0.001,
    "epsilon": 1e-16,
    "n_iter": 30, 
    "unbalance_init": None,
    #"seed": [np.randint() for i in range(5)]
    "seed": [12,68,415,786174,84,687445,2548,798955,124,174685],
    "inner_iter": 20,
    "inner_tol" : 0
})

# XP3: scaling init impact on zero-locking
#variables = dict({
    #"sp": [1e-2],
    #"xp": ["sparse","lowrank"],
    #"scale_init": True,
    #"rank": 4,
    #"e_rank": 6,
    #"dim": 30,
    #"noise": 0.001,
    #"epsilon": 1e-16,
    #"n_iter": 30, 
    #"unbalance_init": [1e-3,0.01,0.1,1,10,100,1000,1000],
    ##"seed": [np.randint() for i in range(5)]
    #"seed": [12,68,415,786174,84]
#})

@run_and_track(algorithm_names=algorithms,name_store=name,**variables)
def run(**v):
    rng = tl.check_random_state(v["seed"])
    facs =  [tl.tensor(tl.abs(rng.randn(v["dim"],v["rank"]))) for i in range(3)]
    # Sparse first factor
    if v["xp"]=="sparse":
        facs[0] = sparsify(facs[0],0.3) 
    cp_true = tl.cp_tensor.CPTensor((None,facs)) # CP tensor
    cp_true.normalize()
    data = cp_true.to_tensor() + v["noise"]*tl.tensor(rng.randn(v["dim"],v["dim"],v["dim"]))
    # start from same init
    init_cp = tl.random.random_cp(3*[v["dim"]],v["e_rank"],random_state=rng)
    init_cp.normalize()

    if v["unbalance_init"]:
        init_cp[1][0] *= v["unbalance_init"]
        init_cp[1][1] *= v["unbalance_init"]
        init_cp[1][2] *= v["unbalance_init"]

    # decomposition
    if v["xp"]=="sparse":
        spvec = [v["sp"],0,0]
        rvec = [0,v["sp"],v["sp"]]
    elif v["xp"]=="lowrank":
        spvec = [0,0,0]
        rvec = [v["sp"],v["sp"],v["sp"]]
    else:
        print("bad xp name")
        return 
    # Scale init
    if v["scale_init"]:
        init_scaled, scale = scale_factors_fro(init_cp,data,spvec,rvec)
        print(f"Optimal init scaling is {scale}")
        # balance 2d because it does not change scale, but converse is not true
        init = balance_factors(init_scaled,spvec,rvec)
    else:
        init = init_cp

    # Compute lambda max ratio heuristic
    #reg_amnt,_ = heuristic_regmax(init,data,spvec)

    # Balanced method 
    out1 = non_negative_parafac_hals(data, rank=v["e_rank"], verbose=verbose, return_errors=True, sparsity_coefficients=spvec, epsilon=v["epsilon"], n_iter_max=v["n_iter"], init=copy(init), ridge_coefficients=rvec, inner_iter_max=v["inner_iter"], inner_tol=v["inner_tol"])
    # No reg
    out2 = non_negative_parafac_hals(data, rank=v["e_rank"], verbose=verbose, return_errors=True, epsilon=v["epsilon"], n_iter_max=v["n_iter"], init=copy(init), inner_iter_max=v["inner_iter"], inner_tol=v["inner_tol"])
    # No balance, scaled balanced init
    out3 = non_negative_parafac_hals(data, rank=v["e_rank"], verbose=verbose, return_errors=True, sparsity_coefficients=spvec, epsilon=v["epsilon"], n_iter_max=v["n_iter"], rescale=False, init=copy(init), ridge_coefficients=rvec, inner_iter_max=v["inner_iter"], inner_tol=v["inner_tol"])
    # Degenerate (no l2), regular init
    out4 = non_negative_parafac_hals(data, rank=v["e_rank"], verbose=verbose, return_errors=True, sparsity_coefficients=spvec, epsilon=v["epsilon"], n_iter_max=v["n_iter"], rescale=False, init=copy(init_cp), pop_l2=True, inner_iter_max=v["inner_iter"], inner_tol=v["inner_tol"])
    # No balance regular init
    out5 = non_negative_parafac_hals(data, rank=v["e_rank"], verbose=verbose, return_errors=True, sparsity_coefficients=spvec, epsilon=v["epsilon"], n_iter_max=v["n_iter"], rescale=False, init=copy(init_cp), ridge_coefficients=rvec, inner_iter_max=v["inner_iter"], inner_tol=v["inner_tol"])

    # printing and storing final errors
    print(f"loss for sparse ridge balanced: {out1[1][-1]}")
    print(f"loss for unregularized : {out2[1][-1]}")
    print(f"loss for sparse ridge no balance ++init: {out3[1][-1]}")
    print(f"loss for degen : {out4[1][-1]}")
    print(f"loss for sparse ridge no balance : {out5[1][-1]}")

    # Computing error metrics
    cp1 = out1[0]
    cp2 = out2[0]
    cp3 = out3[0]
    cp4 = out4[0]
    cp5 = out5[0]

    # plot sparsity level (first factor)
    tol = 2*v["epsilon"]
    spfacs1=tl.sum(cp1[1][0]>tol)
    spfacs2=tl.sum(cp2[1][0]>tol)
    spfacs3=tl.sum(cp3[1][0]>tol)
    spfacs4=tl.sum(cp4[1][0]>tol)
    spfacs5=tl.sum(cp5[1][0]>tol)
    spfacst=tl.sum(facs[0]>0)
    #print(f"Sparsity levels: (sp value: {sp}), rescaled {spfacs1[-1]}, unreg {spfacs2[-1]}, no rescale init++ {spfacs3[-1]}, l1 degen {spfacs4[-1]}, no rescale {spfacs4[-1]}")

    # Evaluate factor match score
    fms1=fms(cp1,cp_true)
    fms2=fms(cp2,cp_true)
    fms3=fms(cp3,cp_true)
    fms4=fms(cp4,cp_true)
    fms5=fms(cp5,cp_true)


    return dict({
        "sparsity": [spfacs1,spfacs2,spfacs3,spfacs4,spfacs5], 
        "fms": [fms1,fms2,fms3,fms4,fms5],
        "true_sparsity": spfacst,
        #"reg_amnt": reg_amnt,
        "final_errors": [out1[1][-1],out2[1][-1],out3[1][-1],out4[1][-1],out5[1][-1]],
        "errors": [out1[1],out2[1],out3[1],out4[1],out5[1]],
    })

#offset = splist[1]
#splist = [elem + offset for elem in splist]
#plt.figure()
#plt.subplot(131)
#plt.title("sparsity of first factor")
#plt.semilogx(splist, spfacs1)
#plt.semilogx(splist, spfacs2)
#plt.semilogx(splist, spfacs3)
#plt.semilogx(splist, spfacs4)
#plt.semilogx(splist, spfacs5)
#plt.semilogx(splist, spfacst)
#plt.xlabel('sparsity +1')
#plt.legend(['ridge scale','no ridge', 'ridge no-scale init++', 'sparse(*) no ridge', 'ridge no-scale', 'true sparsity level'])

#plt.subplot(132)
#plt.title("Final loss values")
#plt.semilogx(splist, lossf1)
#plt.semilogx(splist, lossf2)
#plt.semilogx(splist, lossf3)
#plt.semilogx(splist, lossf4)
#plt.semilogx(splist, lossf5)
#plt.xlabel(f'sparsity + {offset}')
#plt.legend(['ridge scale','no ridge', 'ridge no-scale init++', 'sparse(*) no ridge', 'ridge no-scale'])

#plt.subplot(133)
#plt.title("Final fms values")
#plt.semilogx(splist, fms1)
#plt.semilogx(splist, fms2)
#plt.semilogx(splist, fms3)
#plt.semilogx(splist, fms4)
#plt.semilogx(splist, fms5)
#plt.xlabel(f'sparsity + {offset}')
#plt.legend(['ridge scale','no ridge', 'ridge no-scale init++', 'sparse(*) no ridge', 'ridge no-scale'])
## Bazooka
#plt.show()
