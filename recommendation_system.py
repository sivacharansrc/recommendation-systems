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
users_interactions = local_path + "\\recommendation-system\\input\\users_interactions.csv"

shared_articles = pd.read_csv(shared_articles)
users_interactions = pd.read_csv(users_interactions)