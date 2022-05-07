#%%
import numpy as np
#%%
size = 10
a = np.linspace(0, 1, size)
b = np.linspace(2, 3, size)
c = np.meshgrid(a, b)
c = np.array(c)
# %%
