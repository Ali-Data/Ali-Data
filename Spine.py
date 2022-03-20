# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:33:18 2022

@author: AliKazemnasab
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
df=pd.read_csv('F:\\علم داده _زره ساز/Spine.csv')
summary=df.describe()
np.sum(df.isna())

info=df.info()

df.columns

df.dtypes
df.agg({'PI':['mean','var'],'GS':['mean','std','median']})

df['PI'].agg(['mean','std'])

df['GS'].agg(['mean', 'median','std','var'])

groupby=df.groupby(['Categories']).agg(['mean','std'])


l=[]
for x in np.unique(df['Categories']):
 l.append(df['GS'].loc[df['Categories']==x])
plt.boxplot(l,notch=True)
plt.xticks(np.arange(1,4),np.unique(df['Categories']),rotation=90)
plt.axhline(np.mean(df['GS']),color='red')
#================================================================
l=[df['GS'].loc[df['Categories']==x] for x in df['Categories'].unique()]
box=plt.boxplot(l,notch=True,labels= df['Categories'].unique())
colors = ['cyan', 'lightblue', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_color(color) 





#=============================================================

X=df.iloc[:,:-1]
X=scale(X)
p=PCA()
p.fit(X)
W=p.components_.T
#Get the PC scores based on the centered X 
y=p.fit_transform(X)

pd.DataFrame(W[:,:3],index=df.columns[:-1],columns=['PC1','PC2','PC3'])


#Compute the explained variability by the PC scores
pd.DataFrame(p.explained_variance_ratio_,index=range(1,7),columns=['Explained Variability'])
p.explained_variance_ratio_.cumsum()


pd.DataFrame(p.explained_variance_ratio_.cumsum()
             ,index=np.arange(X.shape[1])+1
             ,columns=['Explained Variability'])




#Get the scree plot
plt.figure(2)
plt.bar(range(1,7),p.explained_variance_,color="blue",edgecolor="Red")

#Get the scatter plot of the first two PC scores
plt.scatter(y[:,0],y[:,1],c="red",marker='o',alpha=0.5)


plt.xlabel('PC Scores 1')
plt.ylabel('PC Scores 2')

for i, Country in enumerate (df['Categories']):
    plt.text(y[:,0][i]+0.05, y[:,1][i]+0.05, Country)
    

Y=p.fit_transform(X)
plt.figure(1)
plt.scatter(Y[:,0],Y[:,1],c="red",marker='o',alpha=0.5)
plt.xlabel('PC Scores 1')
plt.ylabel('PC Scores 2')
xs=Y[:,0]
ys=Y[:,1]
for i in range(len(W[:,0])):
 plt.arrow(np.mean(xs), np.mean(ys), W[i,0]*max(xs), W[i,1]*max(ys),
 color='b', width=0.0005, head_width=0.0025)
 plt.text(W[i,0]*max(xs)+np.mean(xs), +np.mean(ys)+W[i,1]*max(ys),
 list(df.columns.values)[i], color='b')
 
 
 np.where(Y[:,0]>7)
df.iloc[115,:]


    