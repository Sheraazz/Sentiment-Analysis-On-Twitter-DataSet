#!/usr/bin/env python
# coding: utf-8

# #### Reading Data Set

# In[4]:


import pandas as pd
data=pd.read_csv('twitter_training.csv',header=None)
data.head(20)


# finding total null values and deleting them

# In[ ]:


data.isnull().sum()


# #drop null values

# In[24]:


data.dropna(inplace=True)
data.shape


# In[25]:


data.isnull().sum()


# finding value count of each response in column 2

# In[26]:


data[2].value_counts()


# deleting the irrelevent reviews because it will help to find the sentiment

# In[27]:


data_2 = data[data[2]!='Irrelevant']
data_2.shape


# drop column 0 & 1 for now as it will not be helpfull to find sentiment

# In[28]:


data_2.drop(columns=[0,1],inplace=True)
data_2.head()


# printing a value of new data to check how is sentence defined

# In[29]:


data_2.iloc[1000,1]


# importing libraries for data cleaning

# In[30]:


from bs4 import BeautifulSoup 
import re
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# removing not and no from stop words as we need them for our model training
stopwords = stopwords.words("english")
stopwords.remove('not')
stopwords.remove('no')

# intializing method for lemmatizing words
lemmatizer = WordNetLemmatizer()

# now creating funtion to clean our data
def cleaned_review(review):
    # remove any html tags
    new_review = BeautifulSoup(review).get_text()
    
    # remove urls from reviews
    no_urls = new_review.replace('http\S+', '').replace('www\S+', '')
    
    # remove any non-letters
    clean_review = re.sub("[^a-zA-Z]", " ", no_urls)
    
    # convert whole sentence to lowercase and split
    new_words = clean_review.lower().split()
    
    # converting stopwords list to set for faster search
    stops = set(stopwords)
    
    # using stopwords to remove irrelavent words and lemmatizing the final output
    final_words = [lemmatizer.lemmatize(word) for word in new_words if not word in stops]
    
    # return the final result
    return (" ".join(final_words))


# now we will use our funtion to get cleaned data and no. of words

# In[31]:


# now we will use our funtion to get cleaned data and no. of words
data_2['msg'] = data_2[3].apply(lambda x:cleaned_review(x))


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(20, 12))
# finding ideal length to use for further process
data_2['n_words'] = data_2['msg'].apply(lambda x:len(x.split()))
sns.histplot(data = data_2, x='n_words')
plt.show()


# so we see above that max no.of words in each sentence is 40

# In[33]:


max_len = 40


# now we need to one_hot encode the reviews

# In[34]:


sentiment = pd.get_dummies(data_2[2])
data_3 = pd.concat([data_2,sentiment],axis=1)
data_3.head()


# drop columns no longer needed 

# In[35]:


data_3.drop(columns=[2,3,'n_words'],inplace=True)


# In[36]:


data_3


# In[39]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# now we will tokenize the words to sequence so that our model can understand
# we will also pad the sentences with less than 40 words to make size of each sentence equal
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_3['msg'].values)
sequences = tokenizer.texts_to_sequences(data_3['msg'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
X = pad_sequences(sequences, maxlen=max_len, padding='post')


# In[40]:


# lets see how is our output
# here 0's at ending are defined due to the padding
X[1000]


# In[41]:


X


# In[46]:


y=data_3[["Negative","Neutral","Positive"]]
y


# In[47]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=2500)


# In[48]:


X = tfidf.fit_transform(data_3['msg']).toarray()
featureNames = tfidf.get_feature_names()

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20,random_state=42)


# In[57]:


featureNames


# In[49]:



X_train.shape,X_test.shape


# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[51]:


DTC=DecisionTreeClassifier()


# In[52]:


DTC.fit(X_train,y_train)


# In[54]:


DTC.predict(X_test)


# In[56]:


DTC.score(X_test,y_test)

