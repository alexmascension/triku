# 2.1.1
- Fixed error in knn_array calculation for "conn"

# 2.1.0
- Fixed convolution from knn to knn - 1 times.
- Implemented distances and connectivities to create knn_array (temporary). 

# 2.0.1
- Minor import fixes.
- Unnecessary code simplification.

# 2.0.0
## Major changes to the
- Removed background correction. We saw it was unnecessary and it consumed too much time (in 1.3.X was deactivated by default).
- Improved operability with dense and sparse matrices. Now big adatas should run with triku.
- Removed knn/PCA calculation. Now we take information from adata.obsp for kNN matrices, and from adata.var for PCA/related info.
- CLI interactivity and adata returns are removed to make the code simpler. If the user has to run the CLI specifically for the FS step, 
  they will surely be able to integrate it into the pipeline directly loading the adata.
- Added 'name' parameter to save different experiments.

## Minor changes
- Corrected some `None` attributes to their standard `int` value.
- Removed seed parameter because not necessary.
- Get the knn_array from adata.obsp instead of adata.uns
- Get the knn indices from adata.obs['distances'] instead of adata.obsp['connectivities']. Nonzero components of distances are set to knn 
  but the number of nonzero components of 'connectivities' can be >= knn. To cut out to knn we did an argsort, which is now unnecessary. 
  Therefore, this step saves a lot of time and memory.
- Gene names and count matrix variables are linked to their corresponding adata arrays, and are not independent.
- x_conv / y_conv variables are not stored, to save memory. Also, in some cases their computation is unnecessary, so removing them also saves time. 
- Big convolutions fallback to scipy.signal.fftconvolve when the length of the array is bigger than 250. This step considerably reduces computation times.
- Other minor fixes during convolution and emd calculation that improved the time comsumption (e.g. max -> np.max).


# 1.3.1
Minor version fixes

# 1.3.0
Set k to really be k and not k-1, either in convolution and in knn.

# 1.2.0
Updated "emd_X" variables to "triku_X" to adapt other distances in the future.
