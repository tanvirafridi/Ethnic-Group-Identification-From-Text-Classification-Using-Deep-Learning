
from google.colab import drive
drive.mount('/content/drive')

import warnings
warnings.filterwarnings('ignore')

# Importing necessary libraries and functions :
import pandas as pd
import numpy as np
from math import sqrt
import time

# Text processing libraries :
!pip install gensim
import gensim
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize # Tokenizaion
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

# Plotting libraries :
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn :
import sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt

# Importing the dataset :
DATASET_COLUMNS=['Description','Label']
df = pd.read_excel('/content/drive/MyDrive/Thesis2.0/Ethnic Group Identification Update.xlsx', names=DATASET_COLUMNS)

# Display of the random 5 lines :
df.sample(10)

df.columns

df.info()

"""# Removing null"""

df=df.dropna()

print('length of our data is {} reviews'.format(len(df)))

# Checking for Null values :
print("number of missing values in the dataframe is {}".format(np.sum(df.isnull().any(axis=1))))

print('Count of columns in the data is:  ', len(df.columns))
print('Count of rows in the data is:  ', len(df))

# Define the mapping dictionary
label_mapping = {'চাকমা': 0, 'মারমা': 1, 'ত্রিপুরা': 2}

# Map the target column to numerical values
df['Label'] = df['Label'].map(label_mapping)

texts = df['Description'].astype(str)
labels = df['Label'].astype(int)

sorted_counts = df['Label'].value_counts()
print(sorted_counts)

import seaborn as sns
sns.countplot(x='Label',data=df,palette="Blues")
plt.show()

"""# Deep Learning Model

"""

import pandas as pd
import re
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.client import device_lib
tf.test.gpu_device_name()

# Importing the dataset :
DATASET_COLUMNS=['Description','Label']
df = pd.read_excel('/content/drive/MyDrive/Thesis2.0/Ethnic Group Identification Update.xlsx', names=DATASET_COLUMNS)

df.head()

# Converting sentences to string
df['Label'] = df['Label'].astype(str)

# Importing train test splilt library
from sklearn.model_selection import train_test_split

# Train-Test Splitting
train_data, test_data = train_test_split(df, test_size = 0.30)

# Train and test data dimensions
train_data.shape, test_data.shape

"""#Text Preprocessing - NLP



"""

# Importing NLTK Libraries
import nltk
from nltk.corpus import stopwords
from nltk import *

# Declaring function for text preprocessing

def preprocess_text(main_df):
  df_1 = main_df.copy()

  # remove stopwords
  nltk.download('stopwords')
   # Downloading stopwords
  stop = stopwords.words('english')
  df_1['Description'] = df_1['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

  # remove punctuations and convert to lower case
  df_1['Description'] = df_1['Description'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))

  # remove double spaces
  df_1['Description'] = df_1['Description'].apply(lambda x: re.sub(' ', ' ', x))

  return df_1

# Preprocessing training and test data
train_data = preprocess_text(train_data)
test_data = preprocess_text(test_data)

# Verifying text preprocessing
train_data['Description'].head()

"""# Label Encoding"""

# Declaring train labels
train_labels = train_data['Label']
test_labels = test_data['Label']

# Converting labels to numerical features
import numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels = le.transform(train_labels)
test_labels = le.transform(test_labels)

print(le.classes_)
print(np.unique(train_labels, return_counts=True))
print(np.unique(test_labels, return_counts=True))

# Changing labels to categorical features
import numpy as np
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import numpy as np

train_labels = to_categorical(np.asarray(train_labels))
test_labels = to_categorical(np.array(test_labels))

"""## Tokenization"""

from tensorflow.keras.preprocessing.text import Tokenizer

# Defining training parameters
max_sequence_length = 170
max_words = 2500

tokenizer = Tokenizer(num_words = max_words)  # Selects most frequent words
tokenizer.fit_on_texts(train_data.Description)      # Develops internal vocab based on training text
train_sequences = tokenizer.texts_to_sequences(train_data.Description)  # converts text to sequence

test_sequences = tokenizer.texts_to_sequences(test_data.Description)

# Fixing the sequence length
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_data = pad_sequences(train_sequences, maxlen = max_sequence_length)
test_data = pad_sequences(test_sequences, maxlen = max_sequence_length)
train_data.shape, test_data.shape

"""#  Bi-LSTM Model"""

# Model Parameters
embedding_dim = 32

# Importing Libraries

import tensorflow as tf
import sys, os, re, csv, codecs, numpy as np, pandas as pd
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.layers import Dense, Input, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding

embedding_dim = 32

# Model Training
model = Sequential()
model.add(Embedding(max_words,
                   embedding_dim,
                   input_length=max_sequence_length))

# Bidirectional LSTM
model.add(Bidirectional(LSTM(3, return_sequences=True, dropout=0.2, recurrent_dropout=0)))

model.add(GlobalMaxPool1D())

model.add(Dense(3,activation='softmax'))

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='RMSProp', metrics = ['accuracy'])

# declaring weights of product categories
class_weight = {0:3,
                1:3,
                2:3
                }

# training and validating model
history = model.fit(train_data, train_labels, batch_size=32, epochs= 50, class_weight = class_weight, validation_data=(test_data, test_labels))

# Prediction on Test Data
predicted_bi_lstm = model.predict(test_data)
predicted_bi_lstm

"""## Model Performance Attributes"""

import sklearn
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(test_labels, predicted_bi_lstm.round())
print(sklearn.metrics.classification_report(test_labels, predicted_bi_lstm.round()))

def accuracy_plot(history):

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    fig.suptitle('Model Performance with Epochs', fontsize = 16)
    # Subplot 1
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Model Accuracy', fontsize = 14)
    ax[0].set_xlabel('Epochs', fontsize = 12)
    ax[0].set_ylabel('Accuracy', fontsize = 12)
    ax[0].legend(['train', 'validation'], loc='best')

    # Subplot 2
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model Loss', fontsize = 14)
    ax[1].set_xlabel('Epochs', fontsize = 12)
    ax[1].set_ylabel('Loss', fontsize = 12)
    ax[1].legend(['train', 'validation'], loc='best')


accuracy_plot(history)

"""## Confusion Matrix"""

# Declaring function for plotting confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_cm(model, test_data, test_labels):

    products = ['Chakma', 'Tripura', 'Marma']

    # Calculate predictions
    pred = model.predict(test_data)

    # Declaring confusion matrix
    cm = confusion_matrix(np.argmax(np.array(test_labels),axis=1), np.argmax(pred, axis=1))

    # Heat map labels

    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]

    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3,3)

    # Plotting confusion matrix
    plt.figure(figsize=(10,6))

    sns.heatmap(cm, cmap=plt.cm.Blues, annot=labels, annot_kws={"size": 12}, fmt = '',
                xticklabels = products,
                yticklabels = products)

    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8, rotation = 'horizontal')
    plt.title('Confusion Matrix\n', fontsize=19)
    plt.xlabel('Predicted values', fontsize=15)
    plt.ylabel('Actual values', fontsize=15)

plot_cm(model, test_data, test_labels)

accuracy_Bilstm = history.history['accuracy'][-1]
print("Accuracy BiLSTM: ",format(accuracy_Bilstm*100,".2f")+"%")

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
classes=[0, 1, 2]
n_classes=3
y_score = model.predict(test_data)
fpr = dict()
tpr = dict()
roc_auc = dict()
lw=2
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

"""#  LSTM Model"""

embedding_dim = 32

# Model Training
model = Sequential()
model.add(Embedding(max_words,
                   embedding_dim,
                   input_length=max_sequence_length))

# LSTM
model.add(LSTM(3, return_sequences=True, dropout=0.2, recurrent_dropout=0))

model.add(GlobalMaxPool1D())

model.add(Dense(3,activation='softmax'))

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='RMSProp', metrics = ['accuracy'])

# declaring weights of product categories
class_weight = {0:3,
                1:3,
                2:3
                }

# training and validating model
history = model.fit(train_data, train_labels, batch_size=32, epochs= 50, class_weight = class_weight, validation_data=(test_data, test_labels))

# Prediction on Test Data
predicted_lstm = model.predict(test_data)
predicted_lstm

"""## Model Performance Attributes"""

import sklearn
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(test_labels, predicted_lstm.round())
print(sklearn.metrics.classification_report(test_labels, predicted_lstm.round()))

def accuracy_plot(history):

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    fig.suptitle('Model Performance with Epochs', fontsize = 16)
    # Subplot 1
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Model Accuracy', fontsize = 14)
    ax[0].set_xlabel('Epochs', fontsize = 12)
    ax[0].set_ylabel('Accuracy', fontsize = 12)
    ax[0].legend(['train', 'validation'], loc='best')

    # Subplot 2
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model Loss', fontsize = 14)
    ax[1].set_xlabel('Epochs', fontsize = 12)
    ax[1].set_ylabel('Loss', fontsize = 12)
    ax[1].legend(['train', 'validation'], loc='best')


accuracy_plot(history)

"""## Confusion Matrix"""

# Declaring function for plotting confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_cm(model, test_data, test_labels):

    products = ['Chakma', 'Tripura', 'Marma']

    # Calculate predictions
    pred = model.predict(test_data)

    # Declaring confusion matrix
    cm = confusion_matrix(np.argmax(np.array(test_labels),axis=1), np.argmax(pred, axis=1))

    # Heat map labels

    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]

    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3,3)

    # Plotting confusion matrix
    plt.figure(figsize=(10,6))

    sns.heatmap(cm, cmap=plt.cm.Blues, annot=labels, annot_kws={"size": 12}, fmt = '',
                xticklabels = products,
                yticklabels = products)

    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8, rotation = 'horizontal')
    plt.title('Confusion Matrix\n', fontsize=19)
    plt.xlabel('Predicted values', fontsize=15)
    plt.ylabel('Actual values', fontsize=15)

plot_cm(model, test_data, test_labels)

accuracy_lstm = history.history['accuracy'][-1]
print("Accuracy LSTM: ",format(accuracy_lstm*100,".2f")+"%")

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
classes=[0, 1, 2]
n_classes=3
y_score = model.predict(test_data)
fpr = dict()
tpr = dict()
roc_auc = dict()
lw=2
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

"""# CNN"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.losses import categorical_crossentropy

embedding_dim = 32

# Model Training
model = Sequential()
model.add(Embedding(max_words,
                   embedding_dim,
                   input_length=max_sequence_length))

# CNN
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(MaxPooling1D(2))

model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(3,activation="softmax"))

optimizer = Adam(learning_rate=0.000055,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=optimizer,metrics=["accuracy"],loss=categorical_crossentropy)
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='RMSProp', metrics = ['accuracy'])

# training and validating model
history = model.fit(train_data, train_labels, batch_size=48, epochs= 50, validation_data=(test_data, test_labels))

# Prediction on Test Data
predicted_cnn = model.predict(test_data)
predicted_cnn

import sklearn
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(test_labels, predicted_cnn.round())
print(sklearn.metrics.classification_report(test_labels, predicted_cnn.round()))

# Declaring function for plotting confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_cm(model, test_data, test_labels):

    products = ['Chakma', 'Tripura', 'Marma']

    # Calculate predictions
    pred = model.predict(test_data)

    # Declaring confusion matrix
    cm = confusion_matrix(np.argmax(np.array(test_labels),axis=1), np.argmax(pred, axis=1))

    # Heat map labels

    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]

    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3,3)

    # Plotting confusion matrix
    plt.figure(figsize=(10,6))

    sns.heatmap(cm, cmap=plt.cm.Blues, annot=labels, annot_kws={"size": 12}, fmt = '',
                xticklabels = products,
                yticklabels = products)

    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8, rotation = 'horizontal')
    plt.title('Confusion Matrix\n', fontsize=19)
    plt.xlabel('Predicted Values', fontsize=15)
    plt.ylabel('Actual Values', fontsize=15)

plot_cm(model, test_data, test_labels)

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
classes=[0, 1, 2]
n_classes=3
y_score = model.predict(test_data)
fpr = dict()
tpr = dict()
roc_auc = dict()
lw=2
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

accuracy_CNN = history.history['accuracy'][-1]
print("Accuracy CNN: ",format(accuracy_CNN*100,".2f")+"%")

"""**Accuracy Comparison Plot Between Deep Learning Models**"""

import pandas as pd
import seaborn as sns

df = pd.DataFrame(data=
{'Deep Learning':['Bi-LSTM','LSTM', 'CNN'],
'Accuracy':[accuracy_Bilstm*100,accuracy_lstm*100,accuracy_CNN*100]})

plt.figure(figsize=(10,6))
plt.title("Accuracy Comparison Plot Between Deep Learning Models")

p = sns.barplot(x='Deep Learning', y='Accuracy',data=df)
for i in p.containers:
    labels = [f'{v.get_height():0.02f}%' for v in i]
    p.bar_label(i, labels=labels)

