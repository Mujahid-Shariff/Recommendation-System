# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:29:34 2022

@author: admin
"""#Mujahid Shariff
''
import pandas as pd
import numpy as np
df=pd.read_csv("book.csv",encoding='latin1')
df.shape
df.head()

df.sort_values('UserID')
len(df.UserID.unique())


df['BookRating'].value_counts()
df.BookTitle.value_counts()

user_df = df.pivot_table(index='UserID',
                                 columns='BookTitle',
                                 values='BookRating')
user_df #converting the datafile using pivot_table

list(user_df) #listing all the book names

print(user_df) #printing the new data which we converted using pivot table

user_df.fillna(0, inplace=True) #filling blanks as 0 otherwise python will read that as NaN

#cosine system
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df.values,metric='cosine') #user_df is our new dataset on which we will apply cosine.
                                                                  

user_sim #new data set after applying cosine
user_sim_df = pd.DataFrame(user_sim) #converting the data into dataframe
user_sim_df.index   = df.UserID.unique() #when we convert the data into dataframe, 
                                          #rows and colums will be automatically numbered 0,1,2,3 etc 
                                          #here we are replacing those numbers with UserID
user_sim_df.columns = df.UserID.unique()

user_sim_df.to_csv('E:\\DS - ExcelR\\_booknew.csv') #exporting the data to check results of UserID relationships

user_sim_df  #final data with all the relationships with all users

user_sim_df.iloc[:5,:5] #accessing only first five rows and columns to check result

np.fill_diagonal(user_sim, 0) #since diagonal relationship b/w UserID is 1, changing to O
user_sim_df.iloc[0:7, 0:7] #accessing only first 7 rows and columns to check result, if diagonal data is 0

user_sim_df.max() #finding out the maximum values, ratings in our dataset
user_sim_df.idxmax(axis=1)[0:10] #showing relationship with each user, picks up max values = 1


df[(df['UserID']==276729) | (df['UserID']==276726)] #comparing userID 276729 and 276726 and their ratings etc after identifying the user relationships
df[(df['UserID']==276754) | (df['UserID']==276726)]