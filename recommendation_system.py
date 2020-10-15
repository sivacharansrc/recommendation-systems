### IMPORTING REQUIRED LIBRARIES ####
### REFERENCE - https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101

import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
   'BOOKMARK': 3.0, 
   'FOLLOW': 4.0,
   'COMMENT CREATED': 5.0,  
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

# Since one person can interact many ways for a content, let us aggregate all the eventStregth for a given 
# personId and contentId

interactions_full_df = users_interactions.groupby(['personId', 'contentId'])['eventStrength'].sum().reset_index()
interactions_full_df.groupby(pd.cut(interactions_full_df['eventStrength'], bins=6)).size()


### The interactions are very broadly distributed. Let us consider that only a maximum eventStrength of 15 is
# possible (the strength of all intereactions). So, let's replace any eventStrength greater than 15 as 15

interactions_full_df['eventStrength'] = np.where(interactions_full_df['eventStrength'] > 15, 15, interactions_full_df['eventStrength'])
interactions_full_df.groupby(pd.cut(interactions_full_df['eventStrength'], bins=6)).size()
plt.hist(interactions_full_df['eventStrength'], bins=10)
plt.show()

# Let us smoothen the distribution, by taking a log with base 2. Since the minimum value in the distribution is 1, 
# let us add 1 to all values before we take a log with base 2 or else, we might end up with many zeros is the 
# distribution which we do not want it for the eventStrength

interactions_full_df['eventStrength'] = interactions_full_df['eventStrength'].transform(lambda x : math.log(x+1,2))
interactions_full_df.groupby(pd.cut(interactions_full_df['eventStrength'], bins=6)).size()
interactions_full_df.head()

plt.hist(interactions_full_df['eventStrength'], bins=5)
plt.show()

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df, \
   stratify=interactions_full_df['personId'], \
      test_size=0.20, \
         random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))