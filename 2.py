#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
"""
# # DSCI D 699 Independent Study in Data Science Fall 2018
#
# Advisor: Professor David Crandall
#
# Author: Aniruddha M Godbole
# In this program we create complaints language word embeddings.
"""


import fasttext
complaint_word=fasttext.skipgram('/nobackup/agodbole/allproccomplaint.txt','complaint_word',dim=200)

