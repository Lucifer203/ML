import pyprind
import pandas as pd
import os
import sys

basepath = os.path.join(os.path.expanduser("~"),"Downloads","aclImdb")

labels = {'pos':1,'neg':0}
pbar = pyprind.ProgBar(50000,stream=sys.stdout)
df = pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path = os.path.join(basepath,s,l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
                txt = infile.read()
            df = df._append([[txt,labels[l]]],ignore_index=True)
            pbar.update()

df.columns = ['reviews','sentiment']

import numpy as np 
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv',index=False,encoding='utf-8')

df = pd.read_csv('movie_data.csv',encoding='utf-8')
df = df.rename(columns={"0":"review","1":"sentiment"})
df.head(3)