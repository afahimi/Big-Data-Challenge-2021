#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""in this project we are going to make a list of countries based on their highschooler test score and and also obtain 
the GDP of the same contries and we will check if there is a meaingful relation between GDP and the test score
for this purpose we obtain data from OECD and wordbank data"""


# In[203]:


from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy


# In[204]:


df=pd.read_csv("readingtotal.csv")
df1=df[['LOCATION', 'TIME', "Value"]]
df1=df1.rename(columns={'LOCATION': 'country', 'TIME': 'year', 'Value': 'score'})


# In[205]:


df2=df1[df1['year']==2018]
codelist=df2['country']


# In[206]:


#reading GDP information form World Bank data
dfg1=pd.read_csv('GDP2019.csv')


# In[207]:


#extract only data related to year 2018, which is the latest year data
dfg=dfg1[['code','2018']]


# In[208]:


dft = dfg[dfg['code'].isin(codelist)]
dft = dft.rename(columns={"code":"country"})
df2 = df2.reset_index(drop=True)
dft = dft.reset_index(drop=True)
d=pd.merge(dft, df2)
d.sort_values(by=('2018'))


# In[209]:


#here we want to find/confirm the outlier country which is Luxamburg
d2=d[['2018']]
d2.plot(kind='box')
plt.ylabel("GDP per Country")
plt.xlabel("Countries")
plt.title("GDP per Country in 2018")


# In[210]:


#here we drop the outlier from the data
d1=d.drop([26], axis=0)
d1.plot.scatter(x='2018', y='score', logx=True)


# In[211]:


#now read the Mathematics score based based on each country
math=pd.read_csv("mathtotal.csv")
math1=math[["Value","LOCATION","TIME"]]
math1=math1.rename(columns={'LOCATION': 'country', 'TIME': 'year', 'Value': 'score'})


# In[212]:


#read the information related to year 2018 (to corrolate it to the GDP year 2018)
math2=math1[math1['year']==2018]
codelist1=math2['country']


# In[213]:


#find the reading score for the list of availabe countires
dfmath = dfg[dfg['code'].isin(codelist1)]
dfmath = dfmath.rename(columns={"code":"country"})
m=pd.merge(dfmath, math2)
m=m.sort_values(by=('score'))
m = m.reset_index(drop=True)
m.plot.scatter(x='2018', y='score', logx=True)
m1=m.drop([13], axis=0)
m1.plot.scatter(x='2018', y='score', logx=True)


# In[214]:


#reading Science score data
science=pd.read_csv("sciencetest.csv")
science1=science[["Value","LOCATION","TIME"]]
science1=science1.rename(columns={'LOCATION': 'country', 'TIME': 'year', 'Value': 'score'})


# In[215]:


science2=science1[science1['year']==2018]
codelist2=science2['country']


# In[216]:


dfscience = dfg[dfg['code'].isin(codelist2)]
dfscience = dfscience.rename(columns={"code":"country"})
s=pd.merge(dfscience, science2)
s=s.sort_values(by=('score'))
s = s.reset_index(drop=True)
s.plot.scatter(x='2018', y='score', logx=True)
s1=s.drop([12], axis=0)


# In[217]:


#finding correlation table between data and score
s1.corr()


# In[218]:


s1


# In[219]:


#plotting on log scale, make the graph more readable
s1.plot.scatter(x='2018', y='score', logx=True)
m1.plot.scatter(x='2018', y='score', logx=True)
d1.plot.scatter(x='2018', y='score', logx=True)


# In[220]:


#compare the correlation for all three data table (math, science and reading)
plt.scatter(s1[['2018']], s1[['score']], c='b', marker='x', label='Science')
plt.scatter(m1[['2018']], m1[['score']], c='r', marker='s', label='Math')
plt.scatter(d1[['2018']], d1[['score']], c='g', marker='o', label='Reading')

plt.legend(loc='upper left')
plt.title("Pisa Scores vs. GDP")
plt.ylabel("Scores")
plt.xlabel("GDP")


# In[221]:


plt.scatter(np.log10(s1[['2018']]), s1[['score']], c='b', marker='x', label='Science')
plt.scatter(np.log10(m1[['2018']]), m1[['score']], c='r', marker='s', label='Math')
plt.scatter(np.log10(d1[['2018']]), d1[['score']], c='g', marker='o', label='Reading')

plt.legend(loc='upper left')
plt.title("Pisa Scores vs. GDP")
plt.ylabel("Scores")
plt.xlabel("GDP (LOG10)")


# In[222]:


#regression analysis for the data
lsrls= scipy.stats.linregress(np.log10(s1['2018']), s1['score'])
plt.plot(np.log10(s1['2018']), s1['score'], 'o', label='original data')
plt.plot(np.log10(s1['2018']), lsrls.intercept + lsrls.slope*np.log10(s1['2018']), 'r', label='fitted line')
plt.legend()
plt.ylabel("Scores")
plt.xlabel("GDP (LOG10)")
plt.title("Science Scores vs. GDP LRSL")
lsrls


# In[223]:


#regression analysis for the data
lsrlm= scipy.stats.linregress(np.log10(m1['2018']), m1['score'])
plt.plot(np.log10(m1['2018']), m1['score'], 'o', label='original data')
plt.plot(np.log10(m1['2018']), lsrlm.intercept + lsrlm.slope*np.log10(m1['2018']), 'r', label='fitted line')
plt.legend()
plt.ylabel("Scores")
plt.xlabel("GDP (LOG10)")
plt.title("Math Scores vs. GDP LRSL")
lsrlm


# In[224]:


#regression analysis for the data
lsrld= scipy.stats.linregress(np.log10(d1['2018']), d1['score'])
plt.plot(np.log10(d1['2018']), d1['score'], 'o', label='original data')
plt.plot(np.log10(d1['2018']), lsrld.intercept + lsrld.slope*np.log10(d1['2018']), 'r', label='fitted line')
plt.legend()
plt.ylabel("Scores")
plt.xlabel("GDP (LOG10)")
plt.title("Reading Scores vs. GDP LRSL")
lsrld


# In[225]:


s3=s1.copy()
s3['2018']=np.log10(s3['2018'])
s3.corr()


# In[226]:


d3=d1.copy()
d3['2018']=np.log10(d3['2018'])
d3.corr()


# In[227]:


m3=m1.copy()
m3['2018']=np.log10(m3['2018'])
m3.corr()

