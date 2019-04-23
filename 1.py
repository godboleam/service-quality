#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
"""
# # DSCI D 699 Independent Study in Data Science Fall 2018
#
# Advisor: Professor David Crandall
#
# Author: Aniruddha M Godbole
# In this program we pre-process the complaint narrative text in the CFPB dataset
# irrespective of the type of loan (as we are going to use this for creating
word embeddings and not for topic modeling).
"""

import numpy as np
import pandas as pd
import string
import re
import csv
import pickle

inputfile='/nobackup/agodbole/Consumer_Complaints.csv'   
#### Consumer_Complaints.csv is the CFPB Complaints dataset: https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data
outputfile='/nobackup/agodbole/allproccomplaint.txt'

most_punctuation='[!"#@&\()*+,-./:;<=>?[\\]^_`{|}~]' # excluded  ' $ %

expr_num=re.compile('\d+')
expr_X=re.compile('[XX]+')
expr_space=re.compile('\s+')

def preproc_complaint(text):
	text_0=text.lower()
	text_1=re.sub(most_punctuation,'',text_0)
	text_2=re.sub(expr_num,'*',text_1)
	text_3=re.sub(expr_X,'&',text_2)
	text_4=re.sub(expr_space,' ',text_3)
	return text_4

def try_and_preproc_complaint(x):
	try:
		return preproc_complaint(x)
	except:
		return

df_all_complaint=pd.read_csv(inputfile)
df_all_complaint['proctext']=df_all_complaint['Consumer complaint narrative']

#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html
df_all_complaint['proctext'].map(lambda x:try_and_preproc_complaint(x))

df_proc_complaint=df_all_complaint['proctext'].dropna()

df_proc_complaint.to_csv(outputfile,header=False,index=False,encoding='utf-8')

df_all_compwithnarrative=df_all_complaint.dropna(subset=['proctext'])

f=open('compprocdf','wb')
pickle.dump(df_all_compwithnarrative,f)
f.close()
	


