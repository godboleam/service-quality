## Aniruddha M. Godbole and David J. Crandall. 2019. Empowering Borrowers in their choice of Lenders: Decoding Service Quality from Customer Complaints. In ACM Web Science 2019.

##  Authors: Aniruddha M. Godbole and David J. Crandall https://www.cs.indiana.edu/~djcran/
This research was undertaken as DSCI D 699 Independent Study in Data Science (Fall 2018) at 
the School of Informatics, Computing, and Engineering, Indiana University; Advisor: Professor David Crandall

## A. Acknowledgements
1. We acknowledge that one of the authors first heard about the belief regarding the asymmetrical effect of loss/gain associated with pain/joy from A. V. Rajwade (1936-2018). Any errors in this interpretation are solely ours.
2. We acknowledge the help received from Maciej Kula about the appropriate class in the Spotlight library that could be modified for adding more layers to the neural network. 
3. This research was supported in part by Lilly Endowment, Inc., through its support for the Indiana University Pervasive Technology Institute, and in part by the Indiana METACyt Initiative. The Indiana METACyt Initiative at IU was also supported in part by Lilly Endowment, Inc.


## B. Libraries:The following libraries are required for running the code 
(other than the following common libraries: pandas,numpy,csv,os,operator,warnings,json,gzip,pickle,string,re,matplotlib):
torch 0.4.1
scikit-surprise-1.0.6
spotlight 0.1.5
scikit-learn 0.19.1
langdetect 1.0.7
gensim 3.4.0
fasttext 0.8.3
spacy 2.0.16
python-levenshtein
python-twitter
nltk 3.2.5

## C. Data: 
The Consumer Financial Protection Bureau (CFPB) Complaints Dataset is available at https://www.consumerfinance.gov/data-research/consumer-complaints/ ; the World Well-Being Project correlations between each of the five personality traits (on bipolar scales) and words associated with that traits are available here: http://wwbp.org/data.html ; and the TREC 2011 Microblog Dataset which requires an agreement to be signed is available here: https://trec.nist.gov/data/tweets/

## D. The Python code is organized in the following files:
1.py: pre-process the complaint narrative text
2.py: create complaints language word embeddings
3.py: pre-process the English tweets text
4.py: create tweets language word embeddings
4_5.txt: create cross domain word vector space mapping using Vecmap (run at commandline)
5.ipynb: create three topic models for the three types of loans
6A.ipynb: create the proxy trait representative vector
6B.py: compute the 160 scores (150 topic scores and 10 proxy trait scores) for each complaint/complainant.
7.ipynb: Build Complaint Topic Trait Space for the three types of loans using Funk SVD and 
Build Complaint Topic Trait Space for the three types of loans using the Hybrid method. Find top 5 and bottom 5 lenders for an hypothetical user. 
Note: Credentials for using the Twitter API (via python-twitter) will be required for running this Notebook. 

## E. The SAS code is available in the SAScode folder. There is a seprate program for applying the Freeman and Halton Exact test using 
Monte Carlo simulation for the six Complaint Topic Trait Spaces. The SAS output is also available in this folder. The SAS code was run on 
IUAnywhere SAS9.4. The data (RxC contingency table) is embedded in the code. The folder containing the SAS code needs to be designated as 
the IND library. The tests in the case case of mortgage loan Complaint Topic Trait Spaces take around 20 minutes each for Funk SVD and 
Hybrid methods (on IUAnywhere).

## F. Citation Request: If you find anything in this repository helpful please cite our paper: 
Aniruddha M. Godbole and David J. Crandall. 2019. Empowering Borrowers in their choice of Lenders: Decoding Service Quality from Customer Complaints. In ACM Web Science 2019.
