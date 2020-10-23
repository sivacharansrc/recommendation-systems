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
shared_articles = local_path + "\\recommendation-systems\\input\\shared_articles.csv"
shared_articles = pd.read_csv(shared_articles)
shared_articles = shared_articles[shared_articles['eventType'] == 'CONTENT SHARED'].copy().reset_index(drop=True)
# shared_articles.head()

users_interactions = local_path + "\\recommendation-systems\\input\\users_interactions.csv"
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

### FILTERING OFF DATASET THAT HAVE ATLEAST 5 ARTICLES INTERACTED BY A USER

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
# plt.hist(interactions_full_df['eventStrength'], bins=10)
# plt.show()

# Let us smoothen the distribution, by taking a log with base 2. Since the minimum value in the distribution is 1, 
# let us add 1 to all values before we take a log with base 2 or else, we might end up with many zeros, which is the 
# distribution which we do not want it for the eventStrength

interactions_full_df['eventStrength'] = interactions_full_df['eventStrength'].transform(lambda x : math.log(x+1,2))
interactions_full_df.groupby(pd.cut(interactions_full_df['eventStrength'], bins=6)).size()
interactions_full_df.head()

# plt.hist(interactions_full_df['eventStrength'], bins=5)
# plt.show()

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,stratify=interactions_full_df['personId'], test_size=0.20,random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

#Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')

def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:
    # Function to get list of all items not interacted by the user
    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(shared_articles['contentId'])
        non_interacted_items = all_items - interacted_items
        
        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id, interactions_train_indexed_df), topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {
            'hits@5_count':hits_at_5_count,
            'hits@10_count':hits_at_10_count,
            'interacted_count': interacted_items_count_testset,
            'recall@5': recall_at_5,
            'recall@10': recall_at_10}
        return person_metrics
    
    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)
        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {
            'modelName': model.get_model_name(),
            'recall@5': global_recall_at_5,
            'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()

# POPULARITY MODEL - This model acts as a very good baseline, as this model recommends based on overall popularity of the product. In most cases, 
# the recommendations provided are very good, except for the fact that they are not tailored for any individual customer

#Computes the most popular items
item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
item_popularity_df.head(10)

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)].sort_values('eventStrength', ascending = False).head(topn)
        
        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')
            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', left_on = 'contentId', right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]
        return recommendations_df

popularity_model = PopularityRecommender(item_popularity_df, shared_articles)

print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)