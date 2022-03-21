import numpy as np

exact_sv = [1,2,3]
mc_sv = [-1,2,-3]
neyman_sv = [2,3,-4]

error_mc = 0
error_neyman = 0
for i in range(len(exact_sv)):
    error_mc += np.abs(exact_sv[i]-mc_sv[i])
    error_neyman += np.abs(exact_sv[i]-neyman_sv[i])
print("Monte carlo Error: ",error_mc)
print("neyman Error: ",error_neyman)