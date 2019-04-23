#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
"""
# # DSCI D 699 Independent Study in Data Science Fall 2018
#
# Advisor: Professor David Crandall
#
# Author: Aniruddha M Godbole
# In this program we create tweets language word embeddings.
"""

import fasttext
tweet_word=fasttext.skipgram('allengproctweets.txt','tweet_word',dim=200)

