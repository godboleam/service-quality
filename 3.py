#!/usr/bin/env python3
# -*- coding: utf-8 -*-
  
"""
# # DSCI D 699 Independent Study in Data Science Fall 2018
#
# Advisor: Professor David Crandall
#
# Author: Aniruddha M Godbole
# In this program we pre-process the tweets text in the TREC 2011 Microblog dataset.
# A. The input file 'all_tweets' required:
# 1. downloading of 1673 json files (around a 100 files corresponding to specific
# tweets in day during the period 23 Jan 2011 to 08 Feb 2011. Credential provided by
# NIST/TREC 2011 were required to do this download. 
# Twitter Tools (a java program) at https://github.com/lintool/twitter-tools was used to download
# the 1673 files.  
# 2. Then the actual tweets were downloaded using a Python script available at
# https://github.com/cmlonder/trec-collection-downloader 
# This script is extremely useful but is not error free. 
# Overall #1 and #2 took around a week!
#
# B. Around 2/3 of the downloaded tweets are not in English. The langdetect library
# needs time. My analysis is that this was causing a problem when trying to 
# use a lambda function and so this program was re-written to have a loop.

The output file has pre-processed English tweets.
"""
import pandas as pd
import pickle
import string
import re
from langdetect import detect
import time
#import io # for Unix style end of line character

inputfile='all_tweets'
outputfile='allengproctweets.txt'

engprocdf='engprocdf'


most_punctuation='[!"#$%&\()*+,-./:;<=>?[\\]^_`{|}~]' # excluded ' ! @ #  
expr_url=re.compile('http\S+')
expr_num=re.compile('\d+')
expr_handle=re.compile('@\S+') # Email ids too
expr_space=re.compile('\s+')

def preproc_tweet(text):
    text_0=text.lower()
    text_1=re.sub(most_punctuation,'',text_0)
    text_2=re.sub(expr_url,'^',text_1)
    text_3=re.sub(expr_handle,'@',text_2)
    text_4=re.sub(expr_num,'*',text_3)
    text_5=re.sub(expr_space,' ',text_4)
    return text_5


def try_and_preproc_tweet(x):
    try:
        return preproc_tweet(x)
    except:
        return

def try_and_detect(x):
    try:
        return detect(x)
    except:
        return

def main():
    #https://stackoverflow.com/questions/9282967/how-to-open-a-file-using-the-open-with-statement
    with open(inputfile, 'rb') as infile:
        df_all_tweets=pickle.load(infile)
        
    #df_all_tweets['lang'].map(lambda x:try_and_detect(x))
    num_rows=df_all_tweets.shape[0]
        
    with open(outputfile,'w',encoding='utf-8') as outfile:
        
        t1=time.time()
        for row in range(num_rows):
        #for row in range(num_rows):
            text=df_all_tweets['text'].iloc[row]
            if try_and_detect(text)=='en':
                outfile.write(try_and_preproc_tweet(text))
                outfile.write('\n') 
            if row%100000==0:
                print('Number of rows for which pre-processing has been done are:',row)
                t2=time.time()
                print((t2-t1)/(3600),' hours for these many rows.')
    print('Language detection and pre-preocessing done for all (i.e.) ',row,' rows')
    print('Corpus for tweet word vectors is ready')
        

main()
        
        

