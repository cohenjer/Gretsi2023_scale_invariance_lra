import tensorly as tl
from tensorly.decomposition import non_negative_parafac_hals
import numpy as np
import torch as tn
#import tensorly.tenalg as tlg

#for backend in ["numpy","pytorch","mxnet","jax"]:#,"tensorflow","cupy"]:
#    tl.set_backend(backend)
#    tl.tenalg.set_backend('einsum')
rng = tl.check_random_state(hash("hi")%(2**32))
dim = 30
rank = 5
facs =  [tl.tensor(rng.rand(dim,rank)) for i in range(3)]

## -----------------
# Sparse Experiment
## -----------------

for fac in facs:
    fac = tl.clip(fac, a_min=0.3)-0.3
cp_true = tl.cp_tensor.CPTensor((None,facs)) # CP tensor
cp_true.normalize()
data = cp_true.to_tensor() #+ 0.1*tl.tensor(rng.randn(dim,dim,dim))

# decomposition

out = non_negative_parafac_hals(data, rank=rank, verbose=True, return_errors=True, sparsity_coefficients=[0.1,0.1,0], epsilon=1e-8, ridge_coefficients=[0,0,0], n_iter_max=20)
out2 = non_negative_parafac_hals(data, rank=rank, verbose=True, return_errors=True, epsilon=1e-8, n_iter_max=20)

#    cp_out = out[0]
    #cp_out.normalize()
    #fac1 = np.round(cp_out[1][1],5)

    #cp2_out = out2[0]
    #cp2_out.normalize()
    #fac21 = np.round(cp2_out[1][1],5)

    #fac01 = cp_true[1][1]
