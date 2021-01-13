import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
from collections import Counter

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pydot
import graphviz
import json

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def remove_emoji(text):
    """
    a method that removes emojis from text
    :param text: a string, text
    :return:
    """
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_url(text):
    """
    a method that removes url links from text
    :param text:
    :return:
    """
    url_pattern  = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub(r'', text)
 # converting return value from list to string


def clean_text(text):
    """
    a method that cleans the text
    :param text: text or string
    :return:
    """
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    # print('cleaned:'+text1)
    textArr = text1.split()
    text2 = ' '.join([w for w in textArr if (not w.isdigit() and (not w.isdigit() and len(w) > 2))])

    return text2.lower()


def preprocess_df(data):
    """
    a method that performs some further cleaning with dataframe
    :param data: a path to data train or test for tweet sentiment analysis
    :return: cleaned df
    """
    df = pd.read_csv(data)

    df.dropna(axis = 0, how ='any',inplace=True)

    df['Num_words_text'] = df['text'].apply(lambda x:len(str(x).split()))

    mask = df['Num_words_text'] > 2
    train_data = df[mask]

    print('-------Train data--------')
    print(df['sentiment'].value_counts())
    print(len(df))
    print('-------------------------')
    max_sentence_length = train_data['Num_words_text'].max()
    print('Train Max Sentence Length :' + str(max_sentence_length))

    #Apply text preprocessing functions
    df['text'] = df['text'].apply(remove_emoji)
    df['text'] = df['text'].apply(remove_url)
    df['text'] = df['text'].apply(clean_text)

    print(df.head())
    return df, max_sentence_length

train_data, max_train_len = preprocess_df("data/tweet_train.csv")
print()
print("=========================================================================")
print()
test_data, max_test_len = preprocess_df("data/tweet_test.csv")
print("=========================================================================")
print()
print()
print((max_train_len, max_test_len))
max_length = max(max_train_len, max_test_len)

"""https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer 
This class allows to vectorize a text corpus, by turning each text into either a sequence 
of integers (each integer being the index of a token in a dictionary) or into a vector 
where the coefficient for each token could be binary, based on word count, based on tf-idf...
"""

num_words = 20000

#tokenizer = Tokenizer(num_words=num_words, oov_token="unk")
#tokenizer.fit_on_texts(train_data['text'].tolist())

#print(str(tokenizer.texts_to_sequences(['Baby I love you'])))


def tokenizer_to_json(tokenizer):
    json_string = tokenizer.to_json()
    with open('tokenizer.json', 'w') as outfile:
        json.dump(json_string, outfile)

def train_test_data_split_preprocess(max_length):
    """
    a method where train test data were converted to a tf-friendly format
    also a train df was split into train and validation set
    :param max_length: max number of words in a single sentence in data
                        necessary for definition of a common sequence length
                        during padding procedure
    :return:
    """

    X_train, X_valid, y_train, y_valid = train_test_split(train_data['text'].tolist(),\
                                                          train_data['sentiment'].tolist(),\
                                                          test_size=0.1,\
                                                          stratify = train_data['sentiment'].tolist(),\
                                                          random_state=0)


    print('Train data len:'+str(len(X_train)))
    print('Class distribution'+str(Counter(y_train)))
    print('Valid data len:'+str(len(X_valid)))
    print('Class distribution'+ str(Counter(y_valid)))

    tokenizer = Tokenizer(num_words=num_words, oov_token="unk")
    tokenizer.fit_on_texts(train_data['text'].tolist())

    #Tokenize the data
    x_train = np.array( tokenizer.texts_to_sequences(X_train) )
    x_valid = np.array( tokenizer.texts_to_sequences(X_valid) )
    x_test  = np.array( tokenizer.texts_to_sequences(test_data['text'].tolist()) )


    #Create sequences from data sets with the same length
    x_train = pad_sequences(x_train, padding='post', maxlen=max_length)
    x_valid = pad_sequences(x_valid, padding='post', maxlen=max_length)
    x_test = pad_sequences(x_test, padding='post', maxlen=max_length)

    print()
    print()
    print("Example of padded sequence from data:")
    print(x_train[0])
    print()
    print()

    #Perform one-hot encoding for class labels
    le = LabelEncoder()

    train_labels = le.fit_transform(y_train)
    train_labels = np.asarray( tf.keras.utils.to_categorical(train_labels))
    #print(train_labels)
    valid_labels = le.transform(y_valid)
    valid_labels = np.asarray( tf.keras.utils.to_categorical(valid_labels))

    test_labels = le.transform(test_data['sentiment'].tolist())
    test_labels = np.asarray(tf.keras.utils.to_categorical(test_labels))
    list(le.classes_)


    train_ds = tf.data.Dataset.from_tensor_slices((x_train,train_labels))
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid,valid_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test,test_labels))

    #Save the tokenizer to json file

    tokenizer_to_json(tokenizer)

    return train_ds, valid_ds, test_ds


train_ds, valid_ds, test_ds = train_test_data_split_preprocess(max_length)



def CNN(train_ds, valid_ds, embedding_dim, seq_length, max_features, epochs):
    """

    :param embedding_dim: dimension of embedding of each index(word)
    :param seq_length: the maximum length of a tweet in words to be a seq length in CNN
    :param max_features: maximum number of words
    :return:
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(max_features + 1, embedding_dim, input_length=seq_length,\
                                        embeddings_regularizer=regularizers.l2(0.0005)))

    model.add(tf.keras.layers.Conv1D(128, 3, activation='relu',\
                                     kernel_regularizer=regularizers.l2(0.0005),\
                                     bias_regularizer=regularizers.l2(0.0005)))

    model.add(tf.keras.layers.GlobalMaxPooling1D())

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(3, activation='sigmoid',\
                                    kernel_regularizer=regularizers.l2(0.001),\
                                    bias_regularizer=regularizers.l2(0.001), ))

    model.summary()
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='Nadam',
                  metrics=["CategoricalAccuracy"])

    save_model(model, 'CNN_sentiment_analysis')

    # Fit the model using the train and test datasets.
    # history = model.fit(x_train, train_labels,validation_data= (x_test,test_labels),epochs=epochs )
    history = model.fit(train_ds.shuffle(2000).batch(128),
                        epochs=epochs,
                        validation_data=valid_ds.batch(128),
                        verbose=1)

    #tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

    return history

def save_model(model, name):
    model.save(name)

import json

def tokenizer_to_json(tokenizer):
    json_string = tokenizer.to_json()
    with open('tokenizer.json', 'w') as outfile:
        json.dump(json_string, outfile)

"""
history = CNN(train_ds,
    valid_ds,
    embedding_dim=64,
    seq_length=max_length,
    max_features=num_words,
    epochs=30)
"""


def plot_loss(history):
    plt.plot(history.history['loss'], label=' training data')
    plt.plot(history.history['val_loss'], label='validation data)')
    plt.title('Loss for Text Classification')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['categorical_accuracy'], label=' (training data)')
    plt.plot(history.history['val_categorical_accuracy'], label='CategoricalCrossentropy (validation data)')
    plt.title('CategoricalAccuracy for Text Classification')
    plt.ylabel('CategoricalAccuracy value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()


#plot_loss(history)

#plot_accuracy(history)


new_model = tf.keras.models.load_model('CNN_sentiment_analysis')
print(new_model.summary())

with open('tokenizer.json') as json_file:
    json_string = json.load(json_file)
tokenizer1 = tf.keras.preprocessing.text.tokenizer_from_json(json_string)


test_data = preprocess_df("data/tweet_test.csv")

x_test  = np.array( tokenizer1.texts_to_sequences(test_data['text'].tolist()) )
x_test = pad_sequences(x_test, padding='post', maxlen=max_length)

# Generate predictions (probabilities -- the output of the last layer)
# on test  data using `predict`
print("Generate predictions for all samples")
predictions = new_model.predict(x_test)
print(predictions)
predict_results = predictions.argmax(axis=1)

test_data['pred_sentiment']= predict_results
test_data['pred_sentiment'] = np.where((test_data.pred_sentiment == 0),'negative',test_data.pred_sentiment)
test_data['pred_sentiment'] = np.where((test_data.pred_sentiment == '1'),'neutral',test_data.pred_sentiment)
test_data['pred_sentiment'] = np.where((test_data.pred_sentiment == '2'),'positive',test_data.pred_sentiment)

labels = ['positive', 'negative', 'neutral']

print(classification_report(test_data['sentiment'].tolist(), test_data['pred_sentiment'].tolist(), labels=labels))