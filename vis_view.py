import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import TSNE
from os.path import isfile

%matplotlib inline
sns.set(color_codes=True)
#%%

# Load the data
data = []
with open('dataset_m') as rf:
    next(rf)
    for row in rf:
        data.append(tuple(float(x) for x in row.strip().split(' ')))

N = len(data)
data = np.array(data)

X = data[:, 0:108]
y = data[:, 108]
# X_tsne = np.load('xr.npy')

#%%

plt.scatter(X_tsne[y == 0][:, 0], X_tsne[y == 0][:, 1], c='red')
plt.scatter(X_tsne[y == 1][:, 0], X_tsne[y == 1][:, 1], c='blue')

plt.gca().legend(['<=50K', '>50K'])

#%%

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


plt.scatter(X_pca[y == 0][:, 0], X_pca[y == 0][:, 1], c='red')
plt.scatter(X_pca[y == 1][:, 0], X_pca[y == 1][:, 1], c='blue')

plt.gca().legend(['<=50K', '>50K'], loc=3)
