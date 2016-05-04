# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:43:02 2015
Modified on May 3 2016

@author: asangari
@editor: junghwanyang
"""

import json
import re
import numpy as np
import pandas as pd

from nltk import corpus
from nltk.stem import WordNetLemmatizer
from gensim import corpora

from twokenize import tokenizeRawTweetText

from scipy.sparse import lil_matrix

from sklearn.cross_validation import train_test_split
import cPickle as pkl


# Extract tokens from twits and filter them
def tokenize_str(string):
    
    stopwords = corpus.stopwords.words('english')
    stopwords.extend([u"---", u"...", u"n't"])
    lemmatizer = WordNetLemmatizer()
    
    tokens = tokenizeRawTweetText(string)     
    
    tokens = [token.lower() for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token[1:] not in hashtags]
    tokens = [token for token in tokens if token not in urls]
    tokens = [token for token in tokens if token[1:] not in user_mentions]
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [token for token in tokens if not token.isdigit()]

    stemmed_tokens = []
    for w in tokens:
        w = re.sub(r'(.)\1+', r'\1\1', w)
        stemmed_tokens.append(lemmatizer.lemmatize(w)) #.encode("ascii","ignore")
    return stemmed_tokens
    

# Read the labeled data
labeled_datafile = '../R Project/Data/d.2182.csv' #with additional elite codings
labels = pd.read_csv(labeled_datafile, header = 0, encoding = 'latin-1') #changed utf-8 to latin-1


# Read the twitter data raw    
json_file_name = '../R Project/Data/obamacare.json' #find in box folder

text_vocabulary = []
hashtag_vocabulary = []
docs = {}
user_ids = []
user_label = []

line_generator = open(json_file_name,'r')
for line in line_generator:
    
    # Extract different sections of twit from each json entry
    #line_object = json.loads(line.split('\t')[1]) #for 'obamacare sample json.json'
    line_object = json.loads(line) #for 'obamacare.json'
    text = line_object["text"]
    created_at = line_object["created_at"]
    retweeted = line_object["retweeted"]
    retweet_count = line_object["retweet_count"]
    
    twitt_id = line_object["id_str"]
    
    in_reply_to_user_id = line_object["in_reply_to_user_id_str"] #used _str
    in_reply_to_status_id = line_object["in_reply_to_status_id_str"] #used _str
    
    entities = line_object["entities"]
    user_mentions = [h['screen_name'].lower() for h in entities["user_mentions"]]
    hashtags = [h['text'].lower() for h in entities["hashtags"]]
    urls = [h['url'].lower() for h in entities["urls"]]
    
    # User profile
    user = line_object["user"]
    uid = int(user["id"])
    lang = user["lang"]
    description = user["description"]
    location = user["location"]
    geo_enabled = user["geo_enabled"]
    name = user["name"]
    screen_name = user["screen_name"]
    verified = user["verified"]
    
    favourites_count = user["favourites_count"]
    followers_count = user["followers_count"]
    friends_count = user["friends_count"]
    
    
    #Find the label from the labeled file, or enter 0 for unknown ideology
    usr_labels = labels[labels.user_id_str == uid].User_Ideology
 
    if ( (usr_labels == 'C').sum() > (usr_labels == 'L').sum() ):
        user_label.append(-1) # -1 for Conservatives
    elif ( (usr_labels == 'C').sum() < (usr_labels == 'L').sum() ):
        user_label.append(1) # -1 for Liberals
    else:
        user_label.append(0)
        
    #TODO: parse the hashtags and url, and mentions.
    
    text_tokens = tokenize_str(text.lower())
#    print text.lower()    
#    print text_tokens  
#    print '----------------------------------------------'

    if description is not None:
        description_tokens = tokenize_str(description.lower())
        text_tokens.extend(description_tokens)
    
    text_vocabulary.append(text_tokens)   
    
    user_ids.append(int(user['id']))

line_generator.close()     

# Building an array for user ids and user labels
usr_ids = np.array(user_ids)
usr_label = np.array(user_label)


# making a corpus from extracted tokens and building Document Term Matrix (dtm) 
dictionary = corpora.Dictionary(text_vocabulary)
twit_corpus = [dictionary.doc2bow(tweet) for tweet in text_vocabulary]

n_docs = len(text_vocabulary)   
n_terms = len(dictionary.items())
dtm = lil_matrix((n_docs, n_terms))
for doc_i in range(n_docs):
    nnz_i = twit_corpus[doc_i]
    for j in nnz_i:
        dtm[doc_i, j[0]] = j[1]

# Counting the frequency of each term
token_freq = dtm.sum(axis=0)

# Finding the terms with frequency higher than 100
selected_token_ids = (np.array(token_freq[0])[0]>100)

# Limit the tokens to those that are more frequent
selected_token = []
for element in np.where(selected_token_ids)[0]:
    selected_token.append(dictionary.get(element))
DTM = dtm[:,selected_token_ids]


# Extract the labeled samples from DTM and Limit the words to those that are more frequent in the labeled dataset
sub_dtm = DTM[usr_label!=0,:]
labeles = usr_label[usr_label!=0]
token_freq = sub_dtm.sum(axis=0)
selected_token_ids = (np.array(token_freq[0])[0]>100)

selected_words = []
for element in np.where(selected_token_ids)[0]:
    selected_words.append(selected_token[element])

dtm = DTM[:,selected_token_ids]
training_testing_dtm = sub_dtm[:,selected_token_ids]

# Building Testing and Training datasets
x_train, x_test, y_train, y_test = train_test_split(
     training_testing_dtm.toarray(), labeles, test_size=0.5, random_state=42)


# saving features and labels to files
fp = open('./python_files/token_str.pkl', 'wb')
pkl.dump(selected_words, fp)
fp.close()

fp = open('./python_files/training_dataste.pkl', 'wb')
pkl.dump([x_train, x_test, y_train, y_test], fp)
fp.close()

fp = open('./python_files/twitter_dtm.pkl', 'wb')
pkl.dump(dtm, fp)
fp.close()






        #TODO: three approaches: 
        # 1. tokenize, reg_exp to remove punctuations , ... and stem and build dic and word count as features
        
        # 2. PoS tagging and remove some tokens given their PoS and keep the rest in each PoS.
        
        # 3. use RNN embedding on cleaned text (after PoS) as feature.
        
        # 4. Find RNN embeddings of all words, cluster them, centroid as words, BoW.
        
      
      
      # TODO: Find accounts with the most number of hashtags, RT, mentions, followers ,...
