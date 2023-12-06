######## Identifying missing values in tabular data
import pandas as pd
from io import StringIO

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,7.0
10.0,13.0,15.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)
print('\n')
print(df.isnull().sum())  #check for missing values in each coln and return the number of missing values in coln

print(df.values) #access underlying Numpy array of Dataframe

##### Eliminating training examples or features with missing values
# print(df.dropna(axis=0))  #for removing row having Nan values
print(df.dropna(axis=1)) #for removing column with having NaN in any row

print(df.dropna(how='all'))

#drop rows that have fewer than 4 real values
print(df.dropna(thresh=4))

#only drop rows where NaN appear in specific colmn 
print(df.dropna(subset=['C']))

# ### Imputing missing values using mean imputation

from sklearn.impute import SimpleImputer
import numpy as np
imr = SimpleImputer(missing_values=np.nan,strategy='mean') #most_frequent strategy useful for categorical
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

print('\n')
print(df.fillna(df.mean()))

## categorical data encoding with pandas
print('\n')

df = pd.DataFrame([
    ['green','M',10.1,'class2'],
    ['red','L',13.5,'class1'],
    ['blue','XL',15.3,'class2']
])

df.columns = ['color','size','price','classlabel']
print(df)

size_mapping = {'XL':3,'L':2,'M':1}

df['size'] = df['size'].map(size_mapping)
print(df)
print('\n')

# ##inverse mapping int value back to original string
inv_size_mapping = {v: k for k,v in size_mapping.items()} 
df['size'] = df['size'].map(inv_size_mapping)

print(df)

# ## encoding class labels
class_mapping = {label: idx for idx,label in enumerate(np.unique(df['classlabel']))} 
print('\n' , class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print('\n',df)

inv_class_mapping = {v: k for k,v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

print(class_le.inverse_transform(y))

# ### Perfoming one-hot encoding on nominal features
X = df[['color','size','price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print(X)

from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
X = df[['color','size','price']].values
color_ohe = OneHotEncoder()
print(color_ohe.fit_transform(X[:,0].reshape(-1,1)).toarray())


#for applying in multi-feature array

from sklearn.compose import ColumnTransformer
X = df[['color','size','price']].values
c_transf = ColumnTransformer([
    ('color',OneHotEncoder(),[0]),
    ('size',OrdinalEncoder(),[1]),
    ('price','passthrough',[2])])

print(c_transf.fit_transform(X).astype(float))

print(pd.get_dummies(df[['price','color','size']]))

print(pd.get_dummies(df[['price','color','size']],drop_first=True))

color_ohe = OneHotEncoder(categories='auto',drop='first')
c_transf = ColumnTransformer([
    ('color',color_ohe,[0]),
    ('size',OrdinalEncoder(),[1]),
    ('price','passthrough',[2])
])
print(c_transf.fit_transform(X).astype(float))


df = pd.DataFrame([['green','M',10.1,'class2'],
                   ['red','L',13.5,'class1'],
                   ['blue','XL',15.3,'class2']])
df.columns = ['color','size','price','classlabel']
print(df)

df['x>M'] = df['size'].apply(
    lambda x: 1 if x in {'L','XL'} else 0
)
print('\n',df)
df['x>L'] = df['size'].apply(
    lambda x: 1 if x == 'XL' else 0
)
print(df)

del(df['size'])
print(df)


# ###  Partitioning a dataset into separate training and test datasets
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data',header=None)
# print(df_wine.head)

df_wine.columns = ['Class label','Alchol','Malic acid','Ash','Alcalinity of ash'
                   ,'Magnesium','Total phenols','Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins','Color intensity','Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels',np.unique(df_wine['Class label']))
print(df_wine.head())

# ### Partitioning using train_test_split
from sklearn.model_selection import train_test_split
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                 random_state=0,stratify=y) #stratify make sure both training and test have same class proportions

# print(np.unique(y_train).count,np.unique(y_test).count())

## Bringing features onto the same scale
## normalizaiton
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

#### comparison normalization and standardization
ex = np.array([0,1,2,3,4,5])
print('Standardized: ',(ex-ex.mean())/ex.std())

print('Normalized: ',(ex-ex.min())/(ex.max()-ex.min()))

# StandardScaler
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

## L1 and L2 regularization 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1',C=1.0,solver='liblinear',multi_class='ovr')
lr.fit(X_train_std,y_train)
print('Training accuracy: ',lr.score(X_train_std,y_train))
print('Test accuracy: ',lr.score(X_test_std,y_test))
print(lr.intercept_) #this is bias unit
print(lr.coef_) #correspond to weight vector w_j

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors=['blue','green','red','cyan',
        'magenta','yellow','black',
        'pink','lightgreen','lightblue',
        'gray','indigo','orange']
weights,params = [],[]

for c in np.arange(-4.,6.):
    lr=LogisticRegression(penalty='l1',C=10.**c,solver='liblinear'
                          ,multi_class='ovr',random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column,color in zip(range(weights.shape[1]),colors):
    plt.plot(params,weights[:,column],
             label=df_wine.columns[column+1],
             color=color)

plt.axhline(0,color='black',linestyle='--',linewidth=3)
plt.xlim([10**(-5),10**5])
plt.ylabel('Weight coefficient')
plt.xlabel('C (inverse regularization strength)')
plt.xscale('log')
plt.legend(loc='upper left')
# ax.legend(loc='upper center',
#           bbox_to_anchor = (1.38,1.03),
#           ncol=1,fancybox=True)
plt.show()


## #### implementing feature selection algorithm SBS (Sequential Backward Selection)
from sklearn.base import  clone
from itertools import combinations
from sklearn.metrics import accuracy_score


class SBS:
    def __init__(self,estimator,k_features,scoring=accuracy_score,
                 test_size=0.25,random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self,X,y):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=self.test_size,
                                                         random_state=self.random_state)
        
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train,y_train,X_test,y_test,self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores=[]
            subsets=[]

            for p in combinations(self.indices_,r=dim-1):
                score = self._calc_score(X_train,y_train,X_test,y_test,p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self
    
    def transform(self,X):
        return X[:,self.indices_]
    
    def _calc_score(self,X_train,y_train,X_test,y_test,indices):
        self.estimator.fit(X_train[:,indices],y_train)
        y_pred = self.estimator.predict(X_test[:,indices])
        score = self.scoring(y_test,y_pred)
        return score



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn,k_features=1)
sbs.fit(X_train_std,y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

knn.fit(X_train_std[:,k3],y_train)
print('Training accuracy:',knn.score(X_train_std[:,k3],y_train))
print('Test accuracy:',knn.score(X_test_std[:,k3],y_test))

## using randomforest to find the best feature for dimensionality reduction

from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print(indices)

plt.title('Feature importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices],rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest,threshold=0.1,prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold :',X_selected.shape[1])
for f in range(X_selected.shape[1]):
    print('%2d %-*s %f'%(f+1,30,feat_labels[indices[f]],importances[indices[f]]))