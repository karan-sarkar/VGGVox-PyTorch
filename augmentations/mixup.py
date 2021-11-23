import random
import torch
import numpy as np

# def mixup_one_item(spec1,spec2,lbl1,lbl2,alpha = 0.2):
#     lmbda = np.random.beta(alpha,alpha)
#     _,_,col = spec1.shape
#     mix1 = spec1[:,:,:col*lmbda]
#     mix2 = spec2[:,:,col*lmbda+1:]
#     newspec = np.concatenate((mix1,mix2), axis=1)
#     new_label = lbl1 + lbl2
#     new_label = np.where(new_label>1, 1, new_label)
#     return newspec, new_label

def mixup_batch(ds1,ds2,alpha=0.2):
    ex1,lb1 = ds1 #ds is examples and labels
    ex2,lb2 = ds2
    lmbda = np.random.beta(alpha,alpha)
    # _,_,_,col = ex1.shape # assuming shape is (batch_size,layers,rows,columns)
    mix1 = ex1 * lmbda
    mix2 = ex2 * (1 - lmbda)
    # new_ex = np.concatenate((mix1,mix2), axis=1) # concatenate on columns
    new_ex = mix1 + mix2
    new_lbls = lb1 * lmbda + lb2 * (1 - lmbda)
    # new_lbls = new_lbls / np.sum(new_lbls) #combine labels and normalize
    return new_ex, new_lbls
    
    """
    Questions:
        1. How exactly to mixup? Currently just splitting along 
        columns (combining "halves") because it retains enough spec information.
        Should mixup do batch operations?

        2. How to combine labels? Traditional mixup does not make sense, need to just identify
        which speakers are talking. Are currents labels one-hot?

        3. How to test?
    """