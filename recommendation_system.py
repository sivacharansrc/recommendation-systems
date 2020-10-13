### IMPORTING REQUIRED LIBRARIES ####
### REFERENCE - https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101

import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
import os

### READING THE DATA ###

local_path = os.getcwd()
shared_articles = local_path + "\\recommendation-system\\input\\shared_articles.csv"
shared_articles = pd.read_csv(shared_articles)
shared_articles = shared_articles[shared_articles['eventType'] == 'CONTENT SHARED'].copy().reset_index(drop=True)
# shared_articles.head()

users_interactions = local_path + "\\recommendation-system\\input\\users_interactions.csv"
users_interactions = pd.read_csv(users_interactions)
# users_interactions.head()

### ASSOCIATING WEIGHT TO DIFFERENT EVENT TYPE

event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,  
}

users_interactions['eventStrength'] = users_interactions['eventType'].apply(lambda x: event_type_strength[x])

### FILTERING OFF DATASET THAT HAVE ATLEAST 5 ARTICLES READ BY A USER

users_interactions_count_df = users_interactions[['personId', 'contentId']].drop_duplicates().groupby(['personId']).size()
print('Sum of all articles read by users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()['personId'].to_list()
print('Sum of articles read by users with atleast 5 articles each: %d' % len(users_with_enough_interactions_df))

print('Total # of interactions: %d' % users_interactions.shape[0])
users_interactions = users_interactions[users_interactions['personId'].isin(users_with_enough_interactions_df)]
print('Total # of interactions: %d' % users_interactions.shape[0])