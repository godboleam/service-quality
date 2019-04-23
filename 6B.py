#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
"""
# # DSCI D 699 Independent Study in Data Science Fall 2018
#
# Advisor: Professor David Crandall
#
# Author: Aniruddha M Godbole
# In this program the 160 scores (150 scores for the 150 topics and 10 scores for the proxy personality traits were computed. 
#This was computationally intensive as for each complaint for each type of loan each word was considered for computing a similarity 
#with the proxy personality trait vector) and this could not be run 6A.ipynb.
#Please see the output of this program which is available at the end of 6A.ipynb.
# Date 21 October 2018
"""


# In[1]:


#import os
#import csv
import pandas as pd
#import matplotlib.pyplot as plt
import gensim
import numpy as np

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
#from gensim.models.wrappers import LdaMallet
#from gensim.corpora import Dictionary
#import pyLDAvis.gensim

#import os, re, operator, warnings
#warnings.filterwarnings('ignore')  
#get_ipython().run_line_magic('matplotlib', 'inline')
#https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
from numpy.linalg import norm


# In[2]:


import pickle




# In[8]:


#https://github.com/bhargavvader/personal/blob/master/notebooks/text_analysis_tutorial/topic_modelling.ipynb
#https://www.youtube.com/watch?v=ZkAFJwi-G98
import spacy
nlp = spacy.load('en_core_web_sm')


# In[9]:


dfmortproddebtcoll_common_cos_ratings=pd.read_pickle('./data/dfmortproddebtcoll_common_cos_ratings.pickle')
dfstudproddebtcoll_common_cos_ratings=pd.read_pickle('./data/dfstudproddebtcoll_common_cos_ratings.pickle')
dfpayproddebtcoll_common_cos_ratings=pd.read_pickle('./data/dfpayproddebtcoll_common_cos_ratings.pickle')


# In[10]:


model_tweet_to_complaint=gensim.models.KeyedVectors.load_word2vec_format('./data/tweet_complaint_word_vec/tweet_to_complaint.vec')


model_tweet_word=gensim.models.KeyedVectors.load_word2vec_format('./data/tweet_complaint_word_vec/tweet_word.vec')




# In[17]:


model_complaint_to_tweet=gensim.models.KeyedVectors.load_word2vec_format('./data/tweet_complaint_word_vec/complaint_to_tweet.vec')







# In[66]:


def vec_in_cross_domain(word):
    try:
        return model_complaint_to_tweet[word]
    except:
        try:
            return model_tweet_word[word]
        except:
            return np.NaN
        


# In[67]:


def similarity_in_cross_domain(word_1,word_2):
    #checking for the case where a word vector in an array of np.NaN values or the word is a np.NaN value
    if pd.isnull(pd.Series(word_1)).all():
        return np.NaN
    if pd.isnull(pd.Series(word_2)).all():
        return np.NaN
    
    if type(word_1)==np.ndarray:
        word_vector_1=word_1
    else:
        word_vector_1=vec_in_cross_domain(word_1)

    if pd.isnull(pd.Series(word_vector_1)).all():
        return np.NaN
    
    if type(word_2)==np.ndarray:
        word_vector_2=word_2
    else:
        word_vector_2=vec_in_cross_domain(word_2)

    if pd.isnull(pd.Series(word_vector_2)).all():
        return np.NaN        
    
    return(np.dot(word_vector_1,word_vector_2)/(norm(word_vector_1)*norm(word_vector_2)))
    



# In[83]:


def representative_trait_vec(filepath):
    df_trait=pd.read_csv(filepath)
    df_trait['Word Vector']=df_trait['Word'].apply(lambda x:vec_in_cross_domain(x))
    df_trait.dropna(inplace=True)
    trait_corr_sum=df_trait['Correlation'].sum()
    df_trait['Word Vector adj by correlation']=df_trait['Word Vector']*df_trait['Correlation']/trait_corr_sum
    return df_trait['Word Vector adj by correlation'].sum()
    


# In[84]:



# In[87]:


more_agreeable_rep_vec=representative_trait_vec('./data/world_wellbeing_proc/MoreAgreeable.csv')
less_agreeable_rep_vec=representative_trait_vec('./data/world_wellbeing_proc/LessAgreeable.csv')
more_conscientious_rep_vec=representative_trait_vec('./data/world_wellbeing_proc/MoreConscientious.csv')
less_conscientious_rep_vec=representative_trait_vec('./data/world_wellbeing_proc/LessConscientious.csv')
more_open_rep_vec=representative_trait_vec('./data/world_wellbeing_proc/MoreOpen.csv')
less_open_rep_vec=representative_trait_vec('./data/world_wellbeing_proc/LessOpen.csv')
more_extraversion_rep_vec=representative_trait_vec('./data/world_wellbeing_proc/MoreExtraversion.csv')
less_extraversion_rep_vec=representative_trait_vec('./data/world_wellbeing_proc/LessExtraversion.csv')
more_neurotic_rep_vec=representative_trait_vec('./data/world_wellbeing_proc/MoreNeurotic.csv')
less_neurotic_rep_vec=representative_trait_vec('./data/world_wellbeing_proc/LessNeurotic.csv')



# # It seems that the accuracy of the personality trait classification may not be good...but even if it is 60-70% it could still be potentially useful...and if not useful...it will get ignored in the matrix factorization??

# In[117]:


# In[122]:


def complaint_trait_est_score(x,trait_rep_vec):
    sum=0
    word_count=1 #intialized as 1 in order to avoid division be zero problem
    for word in nlp(x):
        if not(pd.isnull(pd.Series(vec_in_cross_domain(word.text))).all()): #This is required so that word not part of the cross domain word vector space does not get considered as np.NaN which then means that for such a single word in the complaint text the entire score become an array of np.NaN values (resulting from the similarity computation) 
            word_trait_est_score=similarity_in_cross_domain(word.text,trait_rep_vec)
            sum=sum+word_trait_est_score
            word_count=word_count+1
    return sum/word_count
        
    


# In[123]:


pd.DataFrame(more_agreeable_rep_vec).to_pickle('./data/more_agreeable_rep_vec.pickle')
pd.DataFrame(less_agreeable_rep_vec).to_pickle('./data/less_agreeable_rep_vec.pickle')
pd.DataFrame(more_conscientious_rep_vec).to_pickle('./data/more_conscientious_rep_vec.pickle')
pd.DataFrame(less_conscientious_rep_vec).to_pickle('./data/less_conscientious_rep_vec.pickle')
pd.DataFrame(more_open_rep_vec).to_pickle('./data/more_open_rep_vec.pickle')
pd.DataFrame(less_open_rep_vec).to_pickle('./data/less_open_rep_vec.pickle')
pd.DataFrame(more_extraversion_rep_vec).to_pickle('./data/more_extraversion_rep_vec.pickle')
pd.DataFrame(less_extraversion_rep_vec).to_pickle('./data/less_extraversion_rep_vec.pickle')
pd.DataFrame(more_neurotic_rep_vec).to_pickle('./data/more_neurotic_rep_vec.pickle')
pd.DataFrame(less_neurotic_rep_vec).to_pickle('./data/less_neurotic_rep_vec.pickle')


# # Note the numpy arrays have been converted into DataFrames and then pickled.

# In[124]:


dfmortproddebtcoll_common_cos_ratings['Less Extraversion']=dfmortproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_extraversion_rep_vec))

print('Hello0')
dfmortproddebtcoll_common_cos_ratings['More Extraversion']=dfmortproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_extraversion_rep_vec))
print('Hello0.5')
dfmortproddebtcoll_common_cos_ratings['More Neuroticism']=dfmortproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_neurotic_rep_vec))

dfmortproddebtcoll_common_cos_ratings['Less Neuroticism']=dfmortproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_neurotic_rep_vec))


dfmortproddebtcoll_common_cos_ratings['More Agreeableness']=dfmortproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_agreeable_rep_vec))

dfmortproddebtcoll_common_cos_ratings['Less Agreeableness']=dfmortproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_agreeable_rep_vec))


dfmortproddebtcoll_common_cos_ratings['More Conscientiousness']=dfmortproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_conscientious_rep_vec))

dfmortproddebtcoll_common_cos_ratings['Less Conscientiousness']=dfmortproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_conscientious_rep_vec))


dfmortproddebtcoll_common_cos_ratings['More Openness']=dfmortproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_open_rep_vec))

dfmortproddebtcoll_common_cos_ratings['Less Openness']=dfmortproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_open_rep_vec))

print('Hello1')
# In[128]:


dfpayproddebtcoll_common_cos_ratings['More Extraversion']=dfpayproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_extraversion_rep_vec))

dfpayproddebtcoll_common_cos_ratings['Less Extraversion']=dfpayproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_extraversion_rep_vec))

dfpayproddebtcoll_common_cos_ratings['More Neuroticism']=dfpayproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_neurotic_rep_vec))

dfpayproddebtcoll_common_cos_ratings['Less Neuroticism']=dfpayproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_neurotic_rep_vec))


dfpayproddebtcoll_common_cos_ratings['More Agreeableness']=dfpayproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_agreeable_rep_vec))

dfpayproddebtcoll_common_cos_ratings['Less Agreeableness']=dfpayproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_agreeable_rep_vec))


dfpayproddebtcoll_common_cos_ratings['More Conscientiousness']=dfpayproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_conscientious_rep_vec))

dfpayproddebtcoll_common_cos_ratings['Less Conscientiousness']=dfpayproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_conscientious_rep_vec))


dfpayproddebtcoll_common_cos_ratings['More Openness']=dfpayproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_open_rep_vec))

dfpayproddebtcoll_common_cos_ratings['Less Openness']=dfpayproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_open_rep_vec))

print('Hello2')
# In[ ]:


dfstudproddebtcoll_common_cos_ratings['More Extraversion']=dfstudproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_extraversion_rep_vec))

dfstudproddebtcoll_common_cos_ratings['Less Extraversion']=dfstudproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_extraversion_rep_vec))

dfstudproddebtcoll_common_cos_ratings['More Neuroticism']=dfstudproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_neurotic_rep_vec))

dfstudproddebtcoll_common_cos_ratings['Less Neuroticism']=dfstudproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_neurotic_rep_vec))


dfstudproddebtcoll_common_cos_ratings['More Agreeableness']=dfstudproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_agreeable_rep_vec))

dfstudproddebtcoll_common_cos_ratings['Less Agreeableness']=dfstudproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_agreeable_rep_vec))


dfstudproddebtcoll_common_cos_ratings['More Conscientiousness']=dfstudproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_conscientious_rep_vec))

dfstudproddebtcoll_common_cos_ratings['Less Conscientiousness']=dfstudproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_conscientious_rep_vec))


dfstudproddebtcoll_common_cos_ratings['More Openness']=dfstudproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,more_open_rep_vec))

dfstudproddebtcoll_common_cos_ratings['Less Openness']=dfstudproddebtcoll_common_cos_ratings['Consumer complaint narrative'].apply(lambda x: complaint_trait_est_score(x,less_open_rep_vec))
print('Hello3')

# In[ ]:


dfmortproddebtcoll_common_cos_ratings_traits=dfmortproddebtcoll_common_cos_ratings
dfpayproddebtcoll_common_cos_ratings_traits=dfpayproddebtcoll_common_cos_ratings
dfstudproddebtcoll_common_cos_ratings_traits=dfstudproddebtcoll_common_cos_ratings


# In[ ]:


dfmortproddebtcoll_common_cos_ratings_traits.to_pickle('./data/dfmortproddebtcoll_common_cos_ratings_traits.pickle')
dfpayproddebtcoll_common_cos_ratings_traits.to_pickle('./data/dfpayproddebtcoll_common_cos_ratings_traits.pickle')
dfstudproddebtcoll_common_cos_ratings_traits.to_pickle('./data/dfstudproddebtcoll_common_cos_ratings_traits.pickle')


# In[ ]:

