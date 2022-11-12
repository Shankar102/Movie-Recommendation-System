#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


credits_df=pd.read_csv('credits.csv')
movies_df=pd.read_csv('movies.csv')


# In[3]:


credits_df.head(5)


# In[4]:


movies_df.head(2)


# In[5]:


movies_df=movies_df.merge(credits_df,on='title')


# In[6]:


movies_df.head(5)


# In[7]:


movies_df.shape


# In[8]:


movies_df.size


# In[9]:


movies_df.isnull().sum()


# In[10]:


movies_df.info()


# In[11]:


movies_df=movies_df[['movie_id','title','overview','cast','crew','genres','keywords']]


# In[12]:


movies_df.head(5)


# In[13]:


movies_df.isnull().sum()


# In[14]:


movies_df.dropna(inplace=True)


# In[15]:


movies_df.head(5)


# In[16]:


movies_df.duplicated().sum()


# In[17]:


movies_df.iloc[0].genres


# In[18]:


import ast


# In[19]:


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[20]:


movies_df['genres']=movies_df['genres'].apply(convert)
movies_df['keywords']=movies_df['keywords'].apply(convert)
movies_df.head(5)


# In[21]:


def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter +=1
        else:
            break
        return L


# In[22]:


movies_df['cast']=movies_df['cast'].apply(convert3)


# In[23]:


movies_df.head(5)


# In[24]:


def fatch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l
        


# In[25]:


movies_df['crew']=movies_df['crew'].apply(fatch_director)


# In[26]:


movies_df.head(5)


# In[27]:


movies_df['overview']=movies_df['overview'].apply(lambda x:x.split())


# In[28]:


movies_df.head(5)


# In[29]:


movies_df['genres']=movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['keywords']=movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
#movies_df['cast']=movies_df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['crew']=movies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[30]:


movies_df.head(5)


# In[31]:


movies_df['tags']=movies_df['overview']+movies_df['genres']+movies_df['keywords']+movies_df['crew']


# In[32]:


movies_df.head(5)


# In[33]:


new_df=movies_df[['movie_id','title','tags']]


# In[34]:


new_df.head(5)


# In[35]:


new_df['tags']=new_df['tags'].apply(lambda x:' '.join(x))


# In[36]:


new_df


# In[37]:


new_df['tags'][0]


# In[38]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[39]:


new_df


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[41]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[42]:


vector=cv.fit_transform(new_df['tags']).toarray()


# In[43]:


vector[0]


# In[44]:


import nltk


# In[47]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[48]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[49]:


new_df['tags']=new_df['tags'].apply(stem)


# In[50]:


from sklearn.metrics.pairwise import cosine_similarity


# In[53]:


cosine_similarity(vector)


# In[54]:


cosine_similarity(vector).shape


# In[55]:


similarity = cosine_similarity(vector)


# In[56]:


similarity[0]


# In[57]:


similarity[0].shape


# In[58]:


sorted(list(enumerate(similarity[0])), reverse=True,key=lambda x:x[1])[1:600]


# In[61]:


def recommend (movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distance=similarity[movie_index]
    movie_list=sorted(list(enumerate(distance)), reverse =True,key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[62]:


recommend('Avatar')


# In[63]:


recommend('Newlyweds')


# In[ ]:




