# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:53:59 2020

@author: Apurva
"""

import numpy as np
import pandas as pd
import wget
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================


# =============================================================================
# Download Data from UCI database, define variable names and replace ? data points as NaN
# =============================================================================
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
wget.download(data_url,'crx.csv')
crx_base = pd.read_csv("crx.csv", names = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"] )
crx_base.replace('?', np.NaN, inplace = True)



# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================


# =============================================================================
# Select few rows from a dataset. top & bottom
# =============================================================================
crx_base.head()
crx_base.tail()
crx_base.head(10)
crx_base.tail(10)
crx_base[:2]
crx_base[-2:]

# =============================================================================
# Variables, Dtype and Missing Count
# =============================================================================
# Object dtype are string variables
# compare the head with dtype, to see if there is any data mismatch
crx_base.info()
crx_base.head()
# String/Object: A1, A4, A5, A6, A7, A9, A10, A12, A13, A14, A16
# Float/Integer: A2, A3, A8, A11, A15
# A14 although consists of only numbers, but appears to be non numerical as has pre 0s appended
# will need to convert the dtype of A2 from object to float
crx_base.A2 = pd.to_numeric(crx_base.A2)
# you can use below codes for alternatively convert features into datetime or string
# pd.to_datetime(crx_base.A2) and crx_base.A2.astype(str)
crx_base.info()
# Counting missing values
crx_base.isnull().sum()

# =============================================================================
# Print summary statistics of all numerical variables
# =============================================================================
crx_base.describe()

# =============================================================================
# Print frequency distribution of selected variables (text, nominal & ordinal)
# =============================================================================
# single variable: string
crx_base.A1.value_counts()
df_o = [x for x in crx_base.columns if crx_base[x].dtype == 'O']
df_n = [x for x in crx_base.columns if crx_base[x].dtype != 'O']
for column in df_o:
    print( crx_base[column].value_counts() )
# single variable: numeric
crx_base.A11.value_counts().sort_index()
# multi variables
pd.crosstab(crx_base.A9, [crx_base.A10, crx_base.A13], margins = True)
pd.crosstab(crx_base.A9, [crx_base.A10, crx_base.A13] , normalize = True )
# we can have percentage view across rows / columns
# pd.crosstab(crx_base.A9, [crx_base.A10, crx_base.A13] , normalize = 'index' )
# pd.crosstab(crx_base.A9, [crx_base.A10, crx_base.A13] , normalize = 'columns' )

# plot a heatmap
plt.figure(figsize=(5, 4))
sns.heatmap( pd.crosstab(crx_base.A9, [crx_base.A10, crx_base.A13] , normalize = True ) ,cmap="YlGnBu", linewidths=.5, annot=True, cbar=True )
bottom, top = plt.ylim()
plt.ylim(bottom + 1.5, top - 1.5)


# =============================================================================
# Sort rows based on few variables.
# =============================================================================
crx_base.sort_values(['A1','A5','A6','A7','A11'],ascending = True, inplace = True)
sort_index.  : for sorting based on index





# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================



# =============================================================================
# Check & remove dataset for number of duplicates
# =============================================================================
# sum duplicates across all variables
crx_base.duplicated( keep = 'last').sum()
# sum duplicates across selected few
crx_base.duplicated( [ 'A6','A7','A13' ], keep = 'last').sum()
# drop duplicate values
crx_base.drop_duplicates( keep = 'first', inplace = True)
# drop duplicate based on select variables
crx_base1 = crx_base.drop_duplicates(['A1','A5','A6','A7','A11'], keep = 'first')

# =============================================================================
# Select few variables from a dataset based on certain where condition.
# =============================================================================

# select & drop few variables
crx_base1 = crx_base[['A1','A2','A3','A4']]
crx_base1 = crx_base.filter( df_o , axis = 1 )
crx_base1 = crx_base.drop( df_n , axis = 1 )
crx_base1 = crx_base.filter(regex = "1")
crx_base1 = crx_base.filter(regex = "1$")crx_base1 = crx_base[df_n].query("A2 > 31 & A11 > 2 ")
crx_base1 = crx_base.filter(regex = "^A1")
crx_base1 = crx_base.filter(regex = "^A[2-5]$")
crx_base1 = crx_base.filter(regex = "2|3")

# loc uses names of columns and rows
# iloc uses position of columns and rows
crx_base1 = crx_base.loc[:,'A5':'A10']
crx_base1 = crx_base.iloc[:,4:10]


# =============================================================================
# Select rows on where condition.
# =============================================================================
# selecting data based on where
crx_base1 = crx_base[crx_base.A1 in ['a','b'] ]
crx_base1 = crx_base[(crx_base.A1 == 'a') | (crx_base.A1 == 'b') ]
crx_base1 = crx_base.query("A1 in ['a','b'] ")
crx_base1 = crx_base.query("A1 == 'a' & A11 == 6 ")


# =============================================================================
# Select few variables from a dataset based on certain where condition.
# =============================================================================





# =============================================================================
# Rename variables
# =============================================================================
crx_base1 = crx_base1.rename(columns = {"A2":"Age","A3":"Debt","A15":"Income"})




# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# =============================================================================
# Select, Where & Groupby: Select few variables based on certain where condition
# and group by elements for various summary statistics (like count sum average etc)
# =============================================================================
# applies to all numerical information, groupers could be numeric / string
crx_base2 = crx_base.groupby(by = ['A7','A6','A11'], as_index = False, sort = False, observed = True).sum()
# count() excludes NAN , size() includes NaN
# group by using pandas methods
crx_base2 = crx_base.groupby(by = ['A1'], as_index = False).agg(mean_A2 = ('A2','mean'), size_A2 = ('A2','size') , count_A2 = ('A2','count') )
# group by using numpy methods
crx_base2 = crx_base.groupby(by = ['A1','A4'], as_index = False).agg({"A2": [np.mean, np.sum], "A3": [np.mean, np.sum], "A8":[np.std , np.size, np.average] })
crx_base2.columns = crx_base2.columns.map('_'.join )

# select, where , group by
crx_base3 = crx_base.query("A10 == 'f'").filter( regex = "^A1[1-5]$" ).query("A12 == 'f'").groupby(by = ['A12','A13'] , as_index = False).agg({"A15":[np.size, len, np.sum, np.mean, np.max]})
crx_base3.columns = crx_base3.columns.map('_'.join )

# we can do similiar manupulation with pivot, but had to rename column and index names
crx_base3 = pd.pivot_table(crx_base, values = ['A15'], index = ['A13'] , columns = ['A12'],  aggfunc = [np.sum, np.mean ] )
crx_base3 = pd.pivot_table(crx_base, values = ['A8','A15'], index = ['A13','A12'] ,  aggfunc = { 'A8': np.sum, 'A15': [min, np.mean ] } )
crx_base3.columns = crx_base3.columns.map('_'.join )
crx_base3.index = crx_base3.index.map('_'.join )
crx_base3['index_A13_A12'] = crx_base3.index



crx_base1 = crx_base.filter(regex = "^A1").drop_duplicates( ['A14','A13'], keep = 'first')
crx_base2 = crx_base[["A11","A14","A2","A3"]].drop_duplicates( ['A14'] , keep = 'first')[:100].rename(columns ={'A14':'B14'})
crx_base2 = crx_base[["A11","A14","A2","A3"]].drop_duplicates( ['A14'] , keep = 'first')[:100]
crx_base1.columns
crx_base2.columns
# =============================================================================
# left outer Join 2 dataset with few variables from both
# =============================================================================
crx_base3 = pd.merge(crx_base1, crx_base2[["B14","A2","A3"]], how = 'left', left_on = 'A14', right_on = 'B14' , indicator = True )
crx_base3 = pd.merge(crx_base1, crx_base2.rename(columns = {"A11":"A11_right"}), how = 'left', on = 'A14', indicator = True )
# validate takes options as '1:1','1:m','m:1','m:m'
crx_base3 = pd.merge(crx_base1, crx_base2, suffixes=('_left', '_right'), how = 'left', on = 'A14', indicator = True, validate = 'm:1' )

# =============================================================================
# Join 2 dataset to keep common rows (Inner)
# =============================================================================
crx_base3 = pd.merge(crx_base1, crx_base[["A14","A2","A3"]], how = 'inner', on = 'A14')
# =============================================================================
# Join multiple dataset
# =============================================================================
crx_base3 = crx_base1.merge(crx_base2[['A14','A2']],on='A14').merge(crx_base2[['A14','A3']],on='A14')

# =============================================================================
# concat 2 dataset, example add 12 monthly data for yearly view.
# =============================================================================
crx_base1 = crx_base[0:200].drop(['A4','A5'], axis = 1)
crx_base2 = crx_base[200:400].drop(['A6','A7'], axis = 1)
crx_base3 = crx_base[400:]
crx_base4 = pd.concat([crx_base1,crx_base2,crx_base3], axis = 0, join = 'inner')
# take just the common columns
crx_base1 = crx_base.iloc[:,0:5]
crx_base2 = crx_base.iloc[:,5:10]
crx_base3 = crx_base.iloc[:,10:]
crx_base4 = pd.concat([crx_base1,crx_base2,crx_base3], axis = 1,keys = ['x','y','z'] )





# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# =============================================================================
# Add new derived variable in data using summary stats on other variables
# =============================================================================
crx_base['A23'] = crx_base['A2'] + crx_base['A3']

# =============================================================================
# Change existing variable based on condition
# =============================================================================
# where ( replcae those values which dont satisfies the condition )
crx_base['A2'] = crx_base['A2'].where( crx_base.A2 < 50, 60)
crx_base.describe()
crx_base = crx_base_copy.copy()
crx_base['A2'] = crx_base['A2'].where( crx_base.A2 < 50, crx_base.A3)
crx_base.describe()


# =============================================================================
# Add new derived variable in data using condition on other variables
# =============================================================================

crx_base['A16N'] = np.where( crx_base['A16'] == '+' , 1, 0 )

crx_base['A16N'] = np.where( crx_base['A2'].isna() , 1, 0 )

crx_base['A16N'] = np.where( crx_base['A16'] == '+' , 1,
                    np.where( crx_base['A4'].isna(), 0, -1 ))


crx_base['A3B'] = pd.cut(crx_base['A3'] , 4, labels = ["W","X","Y","Z"] )

bins = pd.IntervalIndex.from_tuples([(0,1),(1,2),(2,7),(7,28)])
crx_base['A3B'] = pd.cut(crx_base['A3'] , bins)

crx_base['A3D'] = pd.qcut(crx_base['A3'] , 10, labels = [1,2,3,4,5,6,7,8,9,10] )
crx_base['A3Q'] = pd.qcut(crx_base['A3'] , 4 )



bin_dict = {"aa":1,"c":2 ,"cc":2 ,"d":3 ,"e":4 ,"ff":5 ,"i":7 ,"j":7 ,"k":8 ,"m":9 ,"q":10 ,"r":10 ,"w":10 ,"x":10 }
crx_base['A6B'] = crx_base['A6'].map(bin_dict)


# =============================================================================
# Random samples of data
# =============================================================================
crx_base1 = crx_base.sample( n = 100, random_state = 27)
crx_base1 = crx_base.sample( frac = 0.15, random_state = 27)
crx_base2 = crx_base.sample( n = 1000, random_state = 27, replace = True)
crx_base2 = crx_base.sample( frac = 2, random_state = 27, replace = True)

# =============================================================================
# Stratified random sample of data
# =============================================================================
from sklearn.model_selection import train_test_split

crx_base1 , crx_base2  = train_test_split( crx_base, test_size = 0.3, stratify = crx_base["A16N"] , random_state = 27 )

crx_base1["A16N"].value_counts()/len(crx_base1["A16N"])
crx_base2["A16N"].value_counts()/len(crx_base2["A16N"])



# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# =============================================================================
# Calling a piece of code with parameters and output dataset
# =============================================================================


def data_treatment ( base, var_l1, var_l2):
    crx_base1 = base[0:200].drop(var_l1, axis = 1)
    crx_base2 = crx_base[200:400].drop(var_l2, axis = 1)
    crx_base3 = crx_base[400:]
    base_out = pd.concat([crx_base1,crx_base2,crx_base3], axis = 0, join = 'inner')
    return base_out


crx_base4 = data_treatment (crx_base , ['A4','A5'] , ['A6','A7'] )


# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# Creating a new DataFrame
k = np.array(((1,2,3),(2,3,4),(3,4,5)))
crx_base4 = pd.DataFrame(k, columns = ("A","B","C"))

# hard copy vs soft copy
crx_base_copy = crx_base4.copy()
crx_base1 = crx_base4
crx_base.drop(['A4'], inplace = True, axis = 1)


# Length & unique count of a variable
len(crx_base4)
crx_base4.nunique()
crx_base4.A.nunique()

# transpose of a dataframe
crx_base.T
# disolve multi coloumns to just 1
crx_base.melt()

# Viewing largest and smallest values
crx_base.nlargest( 10, 'A3' )
crx_base.nsmallest( 5, 'A8' )

# Missing value treatment
crx_base.isnull().sum()
crx_base.isna().sum()
crx_base_copy = crx_base.copy()
crx_base_copy.fillna(0, inplace = True)
crx_base_copy.isnull().sum()

k = {'A1': 'a' , 'A2':20 , 'A5':'g' , 'A6':'w' , 'A7':'v' , 'A14':'00000' }
crx_base.fillna(value = k, inplace = True)
crx_base.isnull().sum()

# Max Min treatmenab
a = 2.5 * crx_base['A2'].quantile(0.75) - 1.5 * crx_base['A2'].quantile(0.25)
b = 2.5 * crx_base['A2'].quantile(0.25) - 1.5 * crx_base['A2'].quantile(0.75)
crx_base.A2.clip( a , b)



# Plotting variables
crx_base.plot(x = 'A2' , y = 'A3' , kind = 'scatter')
crx_base.plot(y = 'A2'  , kind = 'hist')
crx_base.plot(y = 'A2'  , kind = 'box' )
crx_base.plot(x = 'A2' , y = 'A3' , kind = 'line')




# lambda: 1 line function
k = lambda  x : x.clip( 2.5 * x.quantile(0.75) - 1.5 * x.quantile(0.25) , 2.5 * x.quantile(0.25) - 1.5 * x.quantile(0.75) )
crx_base['A2T'] = k(crx_base['A2'])
crx_base['A3T'] = k(crx_base['A3'])
crx_base['A8T'] = k(crx_base['A8'])
print ( crx_base.describe()[["A2","A2T"]] )

# map: apply a function ( built-in or self defined )  to all elements
print ( list ( map ( len , ['apple' , 'banana' ,'mango' ] ) ) )
k = lambda x : ( x*2)
print ( list ( map ( k , [3,4,5,6,7] ) ) )

k = lambda  x : x.clip( 2.5 * x.quantile(0.75) - 1.5 * x.quantile(0.25) , 2.5 * x.quantile(0.25) - 1.5 * x.quantile(0.75) )
crx_base['A2U'] , crx_base['A3U'], crx_base['A8U']  =   map ( k , ( crx_base['A2'] , crx_base['A3'], crx_base['A8'] ) )
print ( crx_base.describe()[["A2","A2T","A2U"]] )

# apply:
crx_base1 = crx_base[["A2","A3","A8"]].apply(k)

# apply works on a row / column basis of a DataFrame,
# applymap works element-wise on a DataFrame, and
# map works element-wise on a Series.

# filter: applies a condition as a filter on a list and returns only true
print ( list (filter(lambda x: x%2 != 0 , [1,2,3,4,5,6,7,8,9,10] )))



a = ['20','24']
a[1]


pd.to_numeric(crx_base.A2)