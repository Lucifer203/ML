## PCA implementation
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data',header=None)
<<<<<<< HEAD
# print(df_wine.values)
=======
print(df_wine.values)
>>>>>>> 20f2517 (Classification done)

from sklearn.model_selection import train_test_split
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
# print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)

#standardize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigen values\n',eigen_vals)

print(sorted(eigen_vals,reverse=True))

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt
plt.bar(range(1,14),var_exp,align='center',label='Individual explained variance')
plt.step(range(1,14),cum_var_exp,where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('PC index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#Make a list of (eigenvalues,eigenvector) tuples

eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i])
               for i in range(len(eigen_vals))]
# sort the (Evalue,Evector) tuples from hight to low
eigen_pairs.sort(key= lambda k: k[0],reverse=True) #k: k[0] specifies sorting based on eigenValues

w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
print('Matrix W:\n',w)

# print(X_train_std[0].dot(w))

X_train_pca = X_train_std.dot(w)

colors = ['red','blue','green']
markers=['o','s','^']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train==1,0],X_train_pca[y_train==1,1],c=c,label=f'Class {l}',marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#using sklearn

from matplotlib.colors import ListedColormap
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):

    #marker generation and color map
    markers = ('o','s','^','V','<')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors=colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min,x1_max = X[:,0].min() -1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max() +1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    lab = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1,xx2,lab,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
lr=LogisticRegression(multi_class='ovr',random_state=1,solver='lbfgs')

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

#fitting the logistic regression model on reduced dataset
lr.fit(X_train_pca,y_train)
plot_decision_regions(X_train_pca,y_train,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

plot_decision_regions(X_test_pca,y_test,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

loadings = eigen_vecs * np.sqrt(eigen_vals)
fig,ax = plt.subplots()
ax.bar(range(13),loadings[:,0],align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:],rotation=90)
plt.ylim([-1,1])
plt.tight_layout()
plt.show()

sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

fig,ax = plt.subplots()
ax.bar(range(13),sklearn_loadings[:,0],align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:],rotation=90)
plt.ylim([-1,1])
plt.tight_layout()
plt.show()


######### Linear Discriminant Analysis
### Finding the mean vector for (meanx-meany)**2/var**2+var**2
## between class var more within less

np.set_printoptions(precision=4)
mean_vecs =[]
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label],axis=0))
    print(f'MV {label}: {mean_vecs[label-1]}\n')


d = 13
S_W = np.zeros((d,d))
for label , mv in zip(range(1,4),mean_vecs):
    class_scatter = np.zeros((d,d))
    for row in X_train_std[y_train==label]:
        row,mv = row.reshape(d,1),mv.reshape(d,1)
        class_scatter += (row-mv).dot((row-mv).T)
    S_W += class_scatter
print(f'Within class scatter matrix: {S_W.shape[0]}*{S_W.shape[1]}')

print('Class label distribution:',np.bincount(y_train)[1:])
# Class label distribution: [41 50 33]

for label,mv in zip(range(1,4),mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print(f'Within class scatter matrix: {S_W.shape[0]}*{S_W.shape[1]}')

## Between class scatter matrix

mean_overall = np.mean(X_train_std,axis=0)
mean_overall = mean_overall.reshape(d,1)

S_B = np.zeros((d,d))
for i,mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i+1,:].shape[0]
    mean_vec = mean_vec.reshape(d,1)
    S_B += n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)
print(f'Between class scatter matrix: {S_B.shape[0]}*{S_B.shape[1]}')

###### LDA via scikit learn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std,y_train)
lr = LogisticRegression(multi_class='ovr',random_state=1,solver='lbfgs')
lr = lr.fit(X_train_lda,y_train)
plot_decision_regions(X_train_lda,y_train,classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda,y_test,classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# #### Applying non-linear dimensionality reduction using t-distributed stochastic neighbor embedding

from sklearn.datasets import load_digits
digits = load_digits()

fig,ax = plt.subplots(1,4)
for i in range(4):
    ax[i].imshow(digits.images[i],cmap='Greys')
plt.show()

y_digits = digits.target
X_digits = digits.data

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,init='pca',random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)


import matplotlib.patheffects as PathEffects

def plot_projection(x,colors):

    f = plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect='equal')

    for i in range(10):
        plt.scatter(x[colors == i,0],
                    x[colors == i,1])
        
    for i in range(10):
        xtext,ytext = np.median(x[colors == i, :],axis=0)
        txt = ax.text(xtext,ytext,str(i),fontsize = 24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5,foreground='w'),
            PathEffects.Normal()
        ])
        
plot_projection(X_digits_tsne,y_digits)
<<<<<<< HEAD
plt.show()
=======
plt.show()


>>>>>>> 20f2517 (Classification done)
