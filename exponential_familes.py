import matplotlib.pyplot as plt
import numpy as np
import math
#%%
def gaussian_dist(x, mu, std):
    return((1/(np.sqrt(2*np.pi) * std)) * np.exp(-1/2*(((x-mu)/std)**2)))

def expoential_dist(x, theta):
    ys=[]
    for val in x:
        if val>0: ys.append(theta * math.e**(-theta * val))
        else: ys.append(0)
    return(ys)