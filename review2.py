# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 22:31:43 2021

@author: rajam
"""




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



import os
os.chdir(r'C:\\capstone')
os.getcwd()


books = pd.read_csv('pre processed.csv', sep=';',delimiter=',', error_bad_lines=False, encoding="latin-1")


reviews = books["reviews.text"]

import nltk

# Import packages and modules
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# Create a dataframe
X_train = pd.DataFrame(reviews)
X_train.columns = ['textreview']

"""SUCCESS"""
def preprocess_text(text):
    # Tokenise words while ignoring punctuation
    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(text)
    
    # Lowercase and lemmatise 
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
    
    # Remove stopwords
    keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
    return keywords

X_train.loc[:,"textreview"] = X_train.textreview.apply(lambda x : preprocess_text(x))
#preprocess_text(X_train)


# Import module
from nltk.tokenize import RegexpTokenizer
# Create an instance of RegexpTokenizer for alphanumeric tokens
tokeniser = RegexpTokenizer(r'\w+')
# Tokenise 'part1' string
tokens = tokeniser.tokenize(reviews)
print(tokens)


X_train.loc[:,"textreview"] = X_train.textreview.apply(lambda x : str.lower(x))


X_train.describe
X_train.dtypes
X_train['textreview'].astype(str)
X_train.dtypes


X_train['string'] = X_train['textreview'].astype(str)
X_train.dtypes


"""SUCCESS"""
X_train.dropna(axis=0,inplace=True)

"""SUCCESS"""
X_train['textreview'] = pd.Series(X_train['textreview'], dtype="string")
X_train.dtypes
"""SUCCESS"""
X_train.loc[:,"textreview"] = X_train.textreview.apply(lambda x : str.lower(x))
X_train
X_train.loc[:,"textreview"] = X_train.textreview.apply(lambda x : preprocess_text(x))






X_train[pd.notnull(X_train['textreview'])]
X_train['textreview'].dropna
X_train['textreview'].isna


"""SUCCESS"""
X_train.dropna(axis=0,inplace=True)




#df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['sentences']), axis=1)
tokenized = X_train.apply(lambda row: nltk.word_tokenize(X_train['textreview']),axis = 1)


"""
tweetText = X_train['textreview']
from nltk.tokenize import word_tokenize
tweetText = tweetText.apply(word_tokenize)
tweetText.head()
"""


tweetText = X_train['textreview']
from nltk.tokenize import word_tokenize
tweetText = X_train['textreview'].apply(word_tokenize)
tweetText.head()




"""Reference"""

def preprocess_text(text):
    # Tokenise words while ignoring punctuation
    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(text)
    
    # Lowercase and lemmatise 
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
    
    # Remove stopwords
    keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
    return keywords



# Create an instance of TfidfVectorizer
vectoriser = TfidfVectorizer(analyzer=preprocess_text)
# Fit to the data and transform to feature matrix
X_train = vectoriser.fit_transform(X_train['speech'])
# Convert sparse matrix to dataframe
X_train = pd.DataFrame.sparse.from_spmatrix(X_train)
# Save mapping on which index refers to which words
col_map = {v:k for k, v in vectoriser.vocabulary_.items()}
# Rename each column using the mapping
for col in X_train.columns:
    X_train.rename(columns={col: col_map[col]}, inplace=True)
X_train



"""Reference end"""



# Import module
from nltk.stem import WordNetLemmatizer
# Create an instance of WordNetLemmatizer
lemmatiser = WordNetLemmatizer()
# Lowercase and lemmatise tokens
lemmas = [lemmatiser.lemmatize(tweetText) for token in tweetText]
print(lemmas)








import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.metrics


import os
os.chdir(r'C:\\capstone')
os.getcwd()




pd.options.mode.chained_assignment = None 

data = pd.read_csv('pre processed.csv')
data = data.sample(frac=1).reset_index(drop=True)



#data = data[:3000]
print(data.shape)
data.head()
data.describe
data.dtypes

"""
reference 

ratings_explicit = new_ratings[new_ratings.bookRating != 0]
ratings_implicit = new_ratings[new_ratings.bookRating == 0]



users_exp_ratings = users[users.userID.isin(ratings_explicit.userID)]
users_imp_ratings = users[users.userID.isin(ratings_implicit.userID)]

"""

#data.dropna

data = data[['reviews.rating', 'reviews.text']]
data.dtypes
data.head()

print(data.shape)
data.head()
data.describe
data.dtypes




data['reviews.rating'].value_counts().sort_index().plot.bar()

data['reviews.text'].str.len().plot.hist()

data.columns = ['rating','text']
data.dtypes


#ratings_explicit = new_ratings[new_ratings.bookRating != 0]



#data['text'] = data['text'].str.replace('@VirginAmerica', '')
#data.loc[:,"text"] = data['text'].str.replace('@VirginAmerica', '')

data['text'] = pd.Series(data['text'], dtype="string")
#data.dtypes

#X_train.loc[:,"textreview"] = X_train.textreview.apply(lambda x : str.lower(x))

data.dropna(axis=0,inplace=True)
data['rating'].value_counts().sort_index().plot.bar()
data['text'].str.len().plot.hist()

data.shape
data = data[data.text.str.len()!=0]

data = data[data.text.str.len() <=500]
data['rating'].value_counts().sort_index().plot.bar()


#data = ratings_explicit

data['text']=data['text'].apply(lambda x: x.lower())
data['text'].apply(lambda x: x.lower()) #transform text to lowercase
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))
data['text'].head()



#data['text'].dropna

data['rating']=pd.Series(data['rating'],dtype="int")

tokenizer = Tokenizer(num_words=500, split=" ")
tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X) 
# padding our text vector so they all have the same length
X[:5]

#X=X[:1000]


data.reset_index(inplace=True)
data
#ratings_explicit = data



from keras.regularizers import l2

model = Sequential()
model.add(Embedding(500, 90, input_length=X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(90, kernel_regularizer=l2(0.01),recurrent_regularizer=l2(0.01),bias_regularizer=l2(0.01), return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
#model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(90, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))

"""
Different activation functions

sigmoid 
softmax
relu
selu

"""
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall')])
model.summary()


y = pd.get_dummies(data['rating']).values
[print(data['rating'][i], y[i]) for i in range(0,5)]
y
print(data['rating'][0],y[0])
data
data['rating'].value_counts().sort_index().plot.bar()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

y_train_df = pd.DataFrame(y_train, columns =['1','2','3','4','5'])
y_train_df['1'].value_counts().sort_index().plot.bar()
y_train_df['2'].value_counts().sort_index().plot.bar()
y_train_df['3'].value_counts().sort_index().plot.bar()
y_train_df['4'].value_counts().sort_index().plot.bar()
y_train_df['5'].value_counts().sort_index().plot.bar()

y_test_df = pd.DataFrame(y_test, columns =['1','2','3','4','5'])
y_test_df['1'].value_counts().sort_index().plot.bar()
y_test_df['2'].value_counts().sort_index().plot.bar()
y_test_df['3'].value_counts().sort_index().plot.bar()
y_test_df['4'].value_counts().sort_index().plot.bar()
y_test_df['5'].value_counts().sort_index().plot.bar()

batch_size = 32
epochs = 30

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

history.history

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train')
#plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

results = model.evaluate(X_test, y_test, batch_size=32)
print("test loss, test acc, precision, recall:", results)


model.save('sentiment_analysis_30.h5')


predictions = model.predict(X_test)

[print(data['text'][i], predictions[i], y_test[i]) for i in range(0, 5)]


pcount1,pcount2,pcount3,pcount4,pcount5 = 0,0,0,0,0
real1,real2,real3,real4,real5 = 0,0,0,0,0



for i, prediction in enumerate(predictions):
    if np.argmax(prediction)==2:
        pos_count += 1
    elif np.argmax(prediction)==1:
        neu_count += 1
    else:
        neg_count += 1
    
    if np.argmax(y_test[i])==2:
        real_pos += 1
    elif np.argmax(y_test[i])==1:    
        real_neu += 1
    else:
        real_neg +=1


pos_count, neu_count, neg_count = 0, 0, 0
real_pos, real_neu, real_neg = 0, 0, 0
for i, prediction in enumerate(predictions):
    if np.argmax(prediction)==2:
        pos_count += 1
    elif np.argmax(prediction)==1:
        neu_count += 1
    else:
        neg_count += 1
    
    if np.argmax(y_test[i])==2:
        real_pos += 1
    elif np.argmax(y_test[i])==1:    
        real_neu += 1
    else:
        real_neg +=1

print('Positive predictions:', pos_count)
print('Neutral predictions:', neu_count)
print('Negative predictions:', neg_count)
print('Real positive:', real_pos)
print('Real neutral:', real_neu)
print('Real negative:', real_neg)

"""

"""
