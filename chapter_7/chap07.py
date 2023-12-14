## Ensemble methods

from scipy.special import comb
import math

def ensemble_error(n_classifier,error):
    k_start = int(math.ceil(n_classifier/2.))
    probs = [comb(n_classifier,k)*error**k*(1-error)**(n_classifier-k) for k in range(k_start,n_classifier+1)]
    return sum(probs)

print(ensemble_error(n_classifier=11,error=0.25))

import numpy as np
import matplotlib.pyplot as plt

error_range = np.arange(0.0,1.01,0.01)
ens_errors = [ensemble_error(n_classifier=11,error=error) for error in error_range]
plt.plot(error_range,ens_errors,label='Ensemble error',
         linewidth=2)
plt.plot(error_range,error_range,linestyle='--',label='Base error',
         linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()

## weighted majority vote
print(np.argmax(np.bincount([0,0,1],weights=[0.2,0.2,0.6])))

## majority vote for predicting class based on probabilities
ex = np.array([[0.9,0.1],
              [0.8,0.2],
              [0.4,0.6]])
p = np.average(ex,axis=0,weights=[0.2,0.2,0.6]) ## calculates weighted average
print(p)

## MajorityVoteClassifier
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator

class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key,
            value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self,X,y):
        if self.vote not in ('probability','classlabel'):
            raise ValueError(f"vote must be 'Probability'"
                             f"or 'classlabel'"
                             f"; got (vote={self.vote})")
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classifiers and 'f'weights must be equal'f'; got {len(self.weights)} weights,'
                             f'{len(self.classifiers)} classifiers')
        
        # use labelencoder to ensure class labels start
        # with 0, which is important for np.argmax
        #call in self.predict
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    
    def predict(self,X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X),axis=1)
        else:# 'classlabel' vote
            #collect results from clf.predict calls
            predictions = np.asarray([
                clf.predict(X) for clf in self.classifiers_
            ]).T

            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x,weights=self.weights)
                ),
                axis=1,arr=predictions
            )
            maj_vote = self.labelenc_.inverse_transform(maj_vote)
            return maj_vote
        
    def predict_proba(self,X):
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas,axis=0,
                               weights=self.weights)
        return avg_proba
    
    def get_params(self,deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name,step in self.named_classifiers.items():
                for key,value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            
            return out
        
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X,y = iris.data[50:,[1,2]],iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=1,stratify=y)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

clf1 = LogisticRegression(penalty='l2',
                          C=0.001,
                          solver='lbfgs',
                          random_state=1)
clf2 = DecisionTreeClassifier(criterion='gini',max_depth=1,
                              random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')
pipe1 = Pipeline([['sc',StandardScaler()],
                  ['clf',clf1]])
pipe3 = Pipeline([['sc',StandardScaler()],
                  ['clf',clf3]])
clf_labels = ['Logistic regression','Decision tree','KNN']
print('10 fold cross validations:\n')

for clf,label in zip([pipe1,clf2,pipe3],clf_labels):
    scores = cross_val_score(estimator=clf,X=X_train,y=y_train,
                             cv=10,scoring='roc_auc')
    print(f'ROC AUC: {scores.mean():.2f}'
          f'(+/- {scores.std():.2f}) [{label}]')
    
## Now combining individual classifiers for majority rule 
mv_clf = MajorityVoteClassifier(
    classifiers=[pipe1,clf2,pipe3]
)
clf_labels += ['Majority Voting']
all_clf = [pipe1,clf2,pipe3,mv_clf]
for clf,label in zip(all_clf,clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print(f'ROC AUC: {scores.mean():.2f}'
          f'(+/- {scores.std():.2f}) [{label}]')
    
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
colors = ['black','orange','blue','green']
linestyles=[':','--','-.','-']
for clf,label,clr,ls in zip(all_clf,clf_labels,colors,linestyles):
    #assuming label of +ve class is 1
    y_pred = clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
    fpr,tpr,thresholds = roc_curve(y_true=y_test,y_score=y_pred)
    roc_auc = auc(x=fpr,y=tpr)
    plt.plot(fpr,tpr,color=clr,linestyle=ls,label=f'{label} (auc = {roc_auc:.2f})')

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],linestyle='--',
         color='gray',linewidth=2)
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.grid(alpha=0.5)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
# print(mv_clf.get_params())

from sklearn.model_selection import GridSearchCV
params = {'decisiontreeclassifier__max_depth':[1,2],
          'pipeline-1__clf__C':[0.001,0.1,100.0]}
grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring='roc_auc')
grid.fit(X_train,y_train)

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    mean_score = grid.cv_results_['mean_test_score'][r]
    std_dev = grid.cv_results_['std_test_score'][r]
    params = grid.cv_results_['params'][r]
    print(f'{mean_score:.3f} +/- {std_dev:.2f} {params}')

    
    
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/'
                      'wine/wine.data',header=None)
df_wine.columns = ['Class label','Alcohol','Malic acid','Ash','Alcalinity of ash',
                   'Magnesium','Total phenols',
                   'Flavanoids','Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity','Hue','OD280/OD315 of diluted wines',
                   'Proline']
df_wine = df_wine[df_wine['Class label'] !=1]
# print(df_wine)
y = df_wine['Class label'].values
# print(y)
X = df_wine[['Alcohol','OD280/OD315 of diluted wines']].values
# print(X)
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)

from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion='entropy',
                              random_state=1,
                              max_depth=None)
bag = BaggingClassifier(estimator=tree,n_estimators=500,
                        max_features=1.0,max_samples=1.0,bootstrap=True,
                        bootstrap_features=False,n_jobs=1,random_state=1)

from sklearn.metrics import accuracy_score
tree = tree.fit(X_train,y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_true=y_train,y_pred=y_train_pred)
tree_test = accuracy_score(y_true=y_test,y_pred=y_test_pred)
print(f'Decision tree train/test accuracies '
      f'{tree_train:.3f}/{tree_test:.3f}')

bag = bag.fit(X_train,y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train,y_train_pred)
bag_test = accuracy_score(y_test,y_test_pred)
print(f'Bagging  train/test accuracies '
      f'{bag_train:.3f}/{bag_test:.3f}')


x_min = X_train[:,0].min() -1
x_max = X_train[:,0].max() +1
y_min = X_train[:,1].min()-1
y_max = X_train[:,1].max() +1

xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))
f,axarr = plt.subplots(nrows =1,ncols=2,
                       sharex='col',
                       sharey='row',
                       figsize=(8,3))
for idx,clf,tt in zip([0,1],[tree,bag],['Decision tree','Bagging']):
    clf.fit(X_train,y_train)
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx,yy,z,alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0,0],
                       X_train[y_train==0,1],
                       c='blue',marker='^')
    axarr[idx].scatter(X_train[y_train==1,0],
                       X_train[y_train==1,1],
                       c='green',marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol',fontsize=12)
plt.tight_layout()
plt.text(0,-0.2,
         s='OD280/OD315 of diluted wines',
         ha='center',
         va='center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()


# ##### Using Adaptive Boosting for weak learners
y = np.array([1,1,1,-1,-1,-1,1,1,1,-1])
yhat = np.array([1,1,1,-1,-1,-1,-1,-1,-1,-1])

correct = (y == yhat)
weights = np.full(10,0.1)
print(weights)
epsilon = np.mean(~correct) ## ~ invert the incorrect prediction to compute the count of incorrect
print(epsilon)                 ## 1 - correct 0-incorrect
alpha_j = 0.5 * np.log((1-epsilon)/ epsilon)
print(alpha_j)
update_if_correct = 0.1 * np.exp(-alpha_j *1 *1)
print(update_if_correct)

#### increase the weight if predicted incorrectly
update_if_wrong_1 = 0.1 * np.exp(-alpha_j*1*-1)
print(update_if_wrong_1)

weights = np.where(correct ==1,
                   update_if_correct,
                   update_if_wrong_1)
print(weights)

## normalizing the weights so they sum up to 1
normalized_weights = weights/np.sum(weights)
print(normalized_weights)



####### Ada Boost using Scikit Learn
from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion='entropy',
                              random_state=1,
                              max_depth=1)
ada = AdaBoostClassifier(estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=1)
tree = tree.fit(X_train,y_train)
y_train_pred_tree = tree.predict(X_train)
y_test_pred_tree = tree.predict(X_test)
tree_train = accuracy_score(y_train,y_train_pred_tree)
tree_test = accuracy_score(y_test,y_test_pred_tree)
print(f'Decision tree train/test accuracies 'f'{tree_train:.3f}/{tree_test:.3f}')

ada = ada.fit(X_train,y_train)
y_train_pred_ada = ada.predict(X_train)
y_test_pred_ada = ada.predict(X_test)
ada_train = accuracy_score(y_train,y_train_pred_ada)
ada_test = accuracy_score(y_test,y_test_pred_ada)
print(f'AdaBoost train/test accuracies: 'f'{ada_train:.3f}/{ada_test:.3f}')


x_min = X_train[:,0].min()-1
x_max = X_train[:,0].max()+1
y_min = X_train[:,1].min()-1
y_max = X_train[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
f,axarr = plt.subplots(1,2,sharex='col',sharey='row',
                       figsize=(8,3))
for idx,clf,tt in zip([0,1],[tree,ada],['Decision tree','AdaBoost']):
    clf.fit(X_train,y_train)
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx,yy,Z,alpha=0.3)
    axarr[idx].scatter(X_train[y_train == 0,0],
                       X_train[y_train==0,1],
                       c='blue',
                       marker='^')
    axarr[idx].scatter(X_train[y_train==1,0],
                       X_train[y_train==1,1],
                       c='green',
                       marker='o'
                       )
    axarr[idx].set_title(tt)
    axarr[0].set_ylabel('Alcohol',fontsize=12)
plt.tight_layout()
plt.text(0,-0.2,
         s='OD280/OD315 of diluted wines',
         ha='center',
         va='center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()


import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=1000,learning_rate=0.01,
                          max_depth=4,random_state=1,
                          use_label_encoder=False)
gbm = model.fit(X_train,y_train)
y_train_pred_gbm = gbm.predict(X_train)
y_test_pred_gbm = gbm.predict(X_test)
gbm_train = accuracy_score(y_train,y_train_pred_gbm)
gbm_test = accuracy_score(y_test,y_test_pred_gbm)
print(f'XGboost train/test accuracies:'f'{gbm_train:.3f}/{gbm_test:.3f}')
