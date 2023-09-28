#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Write a program that extracts the words (features) used in a sentence.
import pandas as pd
import numpy as np
import re
import nltk


# In[3]:


data = ['The sky is blue and beautiful.',
         'Love this blue and beautiful sky!',
        'The quick brown fox jumps over the lazy dog.',
        'The brown fox is quick and the blue dog is lazy!',
        'The sky is very blue and the sky is very beautiful today',
        'The dog is lazy but the brown fox is quick!' 
]
labels = ['weather', 'weather', 'animals', 'animals', 'weather', 'animals']


# In[4]:


data=np.array(data)


# In[6]:


data_df = pd.DataFrame({'Document': data, 
                         'Category': labels})


# In[7]:


data_df = data_df[['Document', 'Category']]


# In[9]:


data_df


# In[34]:


wpt = nltk.WordPunctTokenizer()
nltk.download("stopwords")
stop_words = nltk.corpus.stopwords.words('english')


# In[35]:


def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc
normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(data)


# In[36]:


norm_corpus


# In[37]:


from nltk.tokenize import word_tokenize
nltk.download('punkt')
tokens_list=[]
for item in norm_corpus:
    nltk_tokens=nltk.word_tokenize(item)
    tokens_list.append(nltk_tokens)
    


# In[33]:


tokens_list


# In[ ]:




