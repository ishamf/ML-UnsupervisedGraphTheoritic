import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import TSNE
from os.path import isfile

# %matplotlib inline
sns.set(color_codes=True)
#%%

# Load the data
data = []
with open('dataset1') as rf:
    next(rf)
    for row in rf:
        data.append(tuple(float(x) for x in row.strip().split(' ')))

N = len(data)
data = np.array(data)

X = data[:, 0:6]
y = data[:, 6]

#%%
Xs = X[np.random.choice(N, size=1000)]
tsne = TSNE(verbose=1)
Xr = tsne.fit_transform(X)

#%%

np.save('xr', Xr)
