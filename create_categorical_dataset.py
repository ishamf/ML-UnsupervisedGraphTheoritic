#%%

import pandas as pd
import numpy as np

#%%

from sklearn import preprocessing

data = pd.read_csv('CensusIncome/CencusIncome.data.txt', encoding='utf8', header=None)
data = data.fillna(data.mean())
data = data.fillna(data.mode(axis=0))

data_t = pd.read_csv('CensusIncome/CencusIncome.test.txt', encoding='utf8', header=None)
data_t = data_t.fillna(data_t.mean())
data_t = data_t.fillna(data_t.mode(axis=0))

#%%

X = data.iloc[:,0:14].values
Xt = data_t.iloc[:,0:14].values
categorical_indices = [1,3,5,6,7,8,9,13]
numeric_indices = [x for x in list(range(14)) if x not in categorical_indices]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for i in categorical_indices:
    labelencoder_X = LabelEncoder()
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    Xt[:, i] = labelencoder_X.transform(Xt[:, i])

#%%

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[:, numeric_indices] = sc_X.fit_transform(X[:, numeric_indices])
Xt[:, numeric_indices] = sc_X.transform(Xt[:, numeric_indices])

onehotencoder_X = OneHotEncoder(categorical_features=categorical_indices)
X = onehotencoder_X.fit_transform(X).toarray()
Xt = onehotencoder_X.transform(Xt).toarray()

#%%

y = data.iloc[:, 14].values
y = y != ' <=50K'
y = y.astype(int)

yt = data_t.iloc[:, 14].values
yt = yt != ' <=50K'
yt = yt.astype(int)

#%%

file = open('dataset_m', 'w')
file.write('{}\n'.format(len(X)))
for i in range(len(X)):
    file.write(' '.join([str(x) for x in X[i,:]]))
    file.write(' {}'.format(y[i]))
    file.write('\n')
file.close()

file = open('dataset_mt', 'w')
file.write('{}\n'.format(len(Xt)))
for i in range(len(Xt)):
    file.write(' '.join([str(x) for x in Xt[i,:]]))
    file.write(' {}'.format(yt[i]))
    file.write('\n')
file.close()

#%%

# Optional
# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(X)

# from matplotlib import pyplot as plt
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y)
# plt.show()
