#!/usr/bin/env python
# coding: utf-8

# In[25]:



import os
import re
import numpy as np
import pandas as pd

# text treatement
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

import warnings
warnings.filterwarnings("ignore")

import gensim
from gensim.models import Word2Vec

import multiprocessing

import csv,sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest,chi2 


# In[2]:


dataset_train = pd.read_csv('project2_training_data.tsv', delimiter = '\t', quoting = 3)
dataset_labels = pd.read_csv('project2_training_data_labels.tsv', delimiter = '\t', quoting = 3)


# In[3]:


dataset_train = dataset_train.dropna(axis='columns', inplace = False)
dataset_labels = dataset_labels.dropna(axis='columns', inplace = False)


# In[4]:


dataset_train.head()


# In[5]:


dataset_labels.head()


# In[6]:


dataset_train.shape


# In[7]:


dataset_labels.shape


# In[8]:


result = pd.concat([dataset_train, dataset_labels], axis=1, join='inner')


# In[9]:


result.head()


# In[10]:


result = result.rename_axis(None, axis=1)


# In[11]:


result.head()


# In[12]:


result.columns = ['Texts', 'Sentiment']


# In[13]:


result.head()


# In[14]:


df = result.copy()


# In[15]:


df = df.dropna(axis='index', inplace=False)


# In[16]:


df.head()


# In[17]:


data_expo = df.copy()


# ## Exploring the distribution
# 

# In[18]:


cnt_pro = data_expo['Sentiment'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Srntiment', fontsize=12)
plt.xticks(rotation=90)
plt.show();


# In[19]:


number_of_words = 0

from bs4 import BeautifulSoup ##code to remove punctuations and symbols
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
data_expo['Texts'] = data_expo['Texts'].apply(cleanText)

for i in range(0,1810):
    lines = data_expo['Texts'][i].split()
    number_of_words += len(lines)
    
print(f"count of words {number_of_words}")

results = set()
unique = data_expo['Texts'].str.lower().str.split().apply(results.update)
number_of_unique_words = len(results)
print(f"count of unique words {number_of_unique_words}")


# ## Cleaning the data for removing punctuations and symbols

# In[20]:


from bs4 import BeautifulSoup
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
df['Texts'] = df['Texts'].apply(cleanText)


# ## The classifier Part

# In[ ]:


opt1=input('Enter\n\t "a" for Classification after tfidf vectorisation \n\t "b" for Classification with Countvectoriser vectorisation \n\t "q" to quit \n')



if opt1=='a':            # simple run with no parameter tuning
#     clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
# #    clf = RandomForestClassifier(criterion='gini',class_weight='balanced') 
    
#     vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,3),token_pattern=r'\b\w+\b')
#     tfidf = vectorizer.fit_transform(data)
#     terms=vectorizer.get_feature_names()
#     tfidf = tfidf.toarray()

#     # Training and Test Split
    
#     trn_data, tst_data, trn_cat, tst_cat = train_test_split(tfidf, labels, test_size=0.20, random_state=42,stratify=labels)   
    
#     #Classificaion    
#     clf.fit(trn_data,trn_cat)
#     predicted = clf.predict(tst_data)
#     predicted =list(predicted)

    data = df['Texts']
    labels = df['Sentiment']

    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
    opt2 = input("Choose a classifier : "
                   "\n\n\t 'lr' to select logistic regression" 
                   "\n\t 'ls' to select Linear SVC" 
                   "\n\t 's' to select SVM" 
                   "\n\t 'dt' to select Decision Tree"   
                   "\n\t 'rf' to select Random Forest"
                   "\n\t 'mn' to select multinomial naive bayes \n\n")   
# Naive Bayes Classifier    
    if opt2=='mn':      
        clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
        clf_parameters = {
        'clf__alpha':(0,1),
        }  
# SVM Classifier
    elif opt2=='ls': 
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,1,2,10,50,100),
        }   
    elif opt2=='s':
        clf = svm.SVC(kernel='linear', class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,0.5,1,2,10,50,100),
        }   
# Logistic Regression Classifier    
    elif opt2=='lr':    
        clf=LogisticRegression(class_weight='balanced') 
        clf_parameters = {
        'clf__solver':('newton-cg','lbfgs','liblinear'),
        }    
# Decision Tree Classifier
    elif opt2=='dt':
        clf = DecisionTreeClassifier(random_state=40)
        clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
        }  
# Random Forest Classifier    
    elif opt2=='rf':
        clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
        clf_parameters = {
                    'clf__criterion':('gini', 'entropy'), 
                    'clf__max_features':('auto', 'sqrt', 'log2'),   
                    'clf__n_estimators':(30,50,100,200),
                    'clf__max_depth':(10,20,),
                    }
        
#     elif opt2 == 'best_classifier':
#         def get_best_classifier(trn_data, trn_cat, for_doc2vec = False):
#             clf = ['LogisticRegression','MultinomialNB','KNeighborsClassifier','DecisionTreeClassifier']
#             parameters = [
#                 {
#                     'clf': LogisticRegression(class_weight='balanced', max_iter=10000),
#                     'clf__solver': ['newton-cg', 'lbfgs', 'liblinear']
#                 },
#                 {
#                     'clf': MultinomialNB(fit_prior=True, class_prior=None) ,
#                     'clf__alpha': [10**-9, 10**-4, 1, 5]
#                 },
#                 {
#                     'clf': KNeighborsClassifier(n_jobs=-1),
#                     'clf__n_neighbors': list(range(1, 20, 4))
#                 },
#                 {
#                     'clf': DecisionTreeClassifier(),
#                     'clf__criterion': ('gini', 'entropy'),
#                     'clf__max_features': ('sqrt', 'log2'),
#                     'clf__ccp_alpha': (0.01, 0.03, 0.05, 0.07, 0.1)
#                 },

#                 {
#                     'clf': svm.SVC(),
#                     'clf__C': [0.1, 1],
#                 }
#             ]

#             if for_doc2vec:
#                 parameters.pop(1)
            
#             max_f1 = 0
#             for params in parameters:
#                         clf = params.pop('clf')
#                         print(f"        -{str(clf)}")

#                         steps = [
#                                 ("scalar", StandardScaler(with_mean=False)),
#                                 ('feature_selector', SelectKBest(chi2, k=1000)),
#                                 ('clf', clf)
#                                 ]
                        
#                         if for_doc2vec:
#                             steps.pop(1)


#                         pipe = Pipeline(steps)
#                         grid_model = GridSearchCV(pipe, param_grid=params, scoring='f1_macro',
#                                       cv=10, n_jobs=-1)
#                         grid_model.fit(X_train, y_train)

#                         print("          -f1_macro Score: ", grid_model.best_score_)
#                         if grid_model.best_score_ > max_f1:
#                                 result = {
#                                             'clf': str(clf),
#                                             'best_score': grid_model.best_score_,
#                                             'best_params': grid_model.best_estimator_,
#                                 }

    else:
        print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
        sys.exit(0)                                  
# Feature Extraction
    pipeline = Pipeline([
    ('vect', TfidfVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k='all')),         
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
    ('clf', clf),]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigra
    }
    
# Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    
    predicted = clf.predict(tst_data)
    predicted =list(predicted)


##################################################### Countvectoriser ###################################################    
    
    

elif opt1=='b':         
    # Training and Test Split
    data = df['Texts']
    labels = df['Sentiment']
    
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
    opt2 = input("Choose a classifier : "
                   "\n\n\t 'lr' to select logistic regression" 
                   "\n\t 'ls' to select Linear SVC" 
                   "\n\t 's' to select SVM" 
                   "\n\t 'dt' to select Decision Tree"   
                   "\n\t 'rf' to select Random Forest"
                   "\n\t 'mn' to select multinomial naive bayes \n\n")   
# Naive Bayes Classifier    
    if opt2=='mn':      
        clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
        clf_parameters = {
        'clf__alpha':(0,1),
        }  
# SVM Classifier
    elif opt2=='ls': 
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,1,2,10,50,100),
        }   
    elif opt2=='s':
        clf = svm.SVC(kernel='linear', class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,0.5,1,2,10,50,100),
        }   
# Logistic Regression Classifier    
    elif opt2=='lr':    
        clf=LogisticRegression(class_weight='balanced') 
        clf_parameters = {
        'clf__solver':('newton-cg','lbfgs','liblinear'),
        }    
# Decision Tree Classifier
    elif opt2=='dt':
        clf = DecisionTreeClassifier(random_state=40)
        clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
        }  
# Random Forest Classifier    
    elif opt2=='rf':
        clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
        clf_parameters = {
                    'clf__criterion':('gini', 'entropy'), 
                    'clf__max_features':('auto', 'sqrt', 'log2'),   
                    'clf__n_estimators':(30,50,100,200),
                    'clf__max_depth':(10,20,),
                    }
        
#     elif opt2 == 'best_classifier':
#         def get_best_classifier(trn_data, trn_cat, for_doc2vec = False):
#             clf = ['LogisticRegression','MultinomialNB','KNeighborsClassifier','DecisionTreeClassifier']
#             parameters = [
#                 {
#                     'clf': LogisticRegression(class_weight='balanced', max_iter=10000),
#                     'clf__solver': ['newton-cg', 'lbfgs', 'liblinear']
#                 },
#                 {
#                     'clf': MultinomialNB(fit_prior=True, class_prior=None) ,
#                     'clf__alpha': [10**-9, 10**-4, 1, 5]
#                 },
#                 {
#                     'clf': KNeighborsClassifier(n_jobs=-1),
#                     'clf__n_neighbors': list(range(1, 20, 4))
#                 },
#                 {
#                     'clf': DecisionTreeClassifier(),
#                     'clf__criterion': ('gini', 'entropy'),
#                     'clf__max_features': ('sqrt', 'log2'),
#                     'clf__ccp_alpha': (0.01, 0.03, 0.05, 0.07, 0.1)
#                 },

#                 {
#                     'clf': svm.SVC(),
#                     'clf__C': [0.1, 1],
#                 }
#             ]

#             if for_doc2vec:
#                 parameters.pop(1)
            
#             max_f1 = 0
#             for params in parameters:
#                         clf = params.pop('clf')
#                         print(f"        -{str(clf)}")

#                         steps = [
#                                 ("scalar", StandardScaler(with_mean=False)),
#                                 ('feature_selector', SelectKBest(chi2, k=1000)),
#                                 ('clf', clf)
#                                 ]
                        
#                         if for_doc2vec:
#                             steps.pop(1)


#                         pipe = Pipeline(steps)
#                         grid_model = GridSearchCV(pipe, param_grid=params, scoring='f1_macro',
#                                       cv=10, n_jobs=-1)
#                         grid_model.fit(X_train, y_train)

#                         print("          -f1_macro Score: ", grid_model.best_score_)
#                         if grid_model.best_score_ > max_f1:
#                                 result = {
#                                             'clf': str(clf),
#                                             'best_score': grid_model.best_score_,
#                                             'best_params': grid_model.best_estimator_,
#                                 }

    else:
        print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
        sys.exit(0)                                  
# Feature Extraction
    pipeline = Pipeline([
    ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k='all')),         
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
    ('clf', clf),]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
    }
    
# Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
     
####################################################### Doc2Vec  ##################################################

# elif opt1=='c':
    
    
    
#     ##building the Doc2vec
    
    
    
#         # Training and Test Split
    
#     train, test = train_test_split(df, test_size=0.25, random_state=42)

    
#     def tokenize_text(text):
#         tokens = []
#         for sent in nltk.sent_tokenize(text):
#             for word in nltk.word_tokenize(sent):
#                 if len(word) < 2:
#                     continue
#                 tokens.append(word.lower())
#         return tokens
#     train_tagged = train.apply(
#         lambda r: TaggedDocument(words=tokenize_text(r['narrative']), tags=[r.Product]), axis=1)
#     test_tagged = test.apply(
#         lambda r: TaggedDocument(words=tokenize_text(r['narrative']), tags=[r.Product]), axis=1)


    
#     model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
#     model_dbow.build_vocab([x for x in tqdm(train_tagged.values)]
    
                          
#     for epoch in range(30):
#         model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
#         model_dmm.alpha -= 0.002
#         model_dmm.min_alpha = model_dmm.alpha 
#         epoch += 1

#     trn_cat, trn_data = vec_for_learning(model_dmm, train_tagged)
#     tst_cat, tst_data = vec_for_learning(model_dmm, test_tagged)

    

    
#     opt2 = input("Choose a classifier : "
#                    "\n\n\t 'lr' to select logistic regression" 
#                    "\n\t 'ls' to select Linear SVC" 
#                    "\n\t 's' to select SVM" 
#                    "\n\t 'dt' to select Decision Tree"   
#                    "\n\t 'rf' to select Random Forest"
#                    "\n\t 'mn' to select multinomial naive bayes \n\n")   
# # Naive Bayes Classifier    
#     if opt2=='mn':      
#         clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
#         clf_parameters = {
#         'clf__alpha':(0,1),
#         }  
# # SVM Classifier
#     elif opt2=='ls': 
#         clf = svm.LinearSVC(class_weight='balanced')  
#         clf_parameters = {
#         'clf__C':(0.1,1,2,10,50,100),
#         }   
#     elif opt2=='s':
#         clf = svm.SVC(kernel='linear', class_weight='balanced')  
#         clf_parameters = {
#         'clf__C':(0.1,0.5,1,2,10,50,100),
#         }   
# # Logistic Regression Classifier    
#     elif opt2=='lr':    
#         clf=LogisticRegression(class_weight='balanced') 
#         clf_parameters = {
#         'clf__solver':('newton-cg','lbfgs','liblinear'),
#         }    
# # Decision Tree Classifier
#     elif opt2=='dt':
#         clf = DecisionTreeClassifier(random_state=40)
#         clf_parameters = {
#         'clf__criterion':('gini', 'entropy'), 
#         'clf__max_features':('auto', 'sqrt', 'log2'),
#         'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
#         }  
# # Random Forest Classifier    
#     elif opt2=='rf':
#         clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
#         clf_parameters = {
#                     'clf__criterion':('gini', 'entropy'), 
#                     'clf__max_features':('auto', 'sqrt', 'log2'),   
#                     'clf__n_estimators':(30,50,100,200),
#                     'clf__max_depth':(10,20,),
#                     }
        
# #     elif opt2 == 'best_classifier':
# #         def get_best_classifier(trn_data, trn_cat, for_doc2vec = False):
# #             clf = ['LogisticRegression','MultinomialNB','KNeighborsClassifier','DecisionTreeClassifier']
# #             parameters = [
# #                 {
# #                     'clf': LogisticRegression(class_weight='balanced', max_iter=10000),
# #                     'clf__solver': ['newton-cg', 'lbfgs', 'liblinear']
# #                 },
# #                 {
# #                     'clf': MultinomialNB(fit_prior=True, class_prior=None) ,
# #                     'clf__alpha': [10**-9, 10**-4, 1, 5]
# #                 },
# #                 {
# #                     'clf': KNeighborsClassifier(n_jobs=-1),
# #                     'clf__n_neighbors': list(range(1, 20, 4))
# #                 },
# #                 {
# #                     'clf': DecisionTreeClassifier(),
# #                     'clf__criterion': ('gini', 'entropy'),
# #                     'clf__max_features': ('sqrt', 'log2'),
# #                     'clf__ccp_alpha': (0.01, 0.03, 0.05, 0.07, 0.1)
# #                 },

# #                 {
# #                     'clf': svm.SVC(),
# #                     'clf__C': [0.1, 1],
# #                 }
# #             ]

# #             if for_doc2vec:
# #                 parameters.pop(1)
            
# #             max_f1 = 0
# #             for params in parameters:
# #                         clf = params.pop('clf')
# #                         print(f"        -{str(clf)}")

# #                         steps = [
# #                                 ("scalar", StandardScaler(with_mean=False)),
# #                                 ('feature_selector', SelectKBest(chi2, k=1000)),
# #                                 ('clf', clf)
# #                                 ]
                        
# #                         if for_doc2vec:
# #                             steps.pop(1)


# #                         pipe = Pipeline(steps)
# #                         grid_model = GridSearchCV(pipe, param_grid=params, scoring='f1_macro',
# #                                       cv=10, n_jobs=-1)
# #                         grid_model.fit(X_train, y_train)

# #                         print("          -f1_macro Score: ", grid_model.best_score_)
# #                         if grid_model.best_score_ > max_f1:
# #                                 result = {
# #                                             'clf': str(clf),
# #                                             'best_score': grid_model.best_score_,
# #                                             'best_params': grid_model.best_estimator_,
# #                                 }

#     else:
#         print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
#         sys.exit(0)                                  
# # Feature Extraction
#     pipeline = Pipeline([
#     ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
#     ('feature_selector', SelectKBest(chi2, k='all')),         
#     ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
#     ('clf', clf),]) 
        
#     feature_parameters = {
#     'vect__min_df': (2,3),
#     'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
#     }
    
# # Classificaion
#     parameters={**feature_parameters,**clf_parameters} 
#     grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
#     grid.fit(trn_data,trn_cat)     
#     clf= grid.best_estimator_  
#     print('********* Best Set of Parameters ********* \n\n')
#     print(clf)
    
#     predicted = clf.predict(tst_data)
#     predicted =list(predicted)
    

############################################################ Remaining options #########################################
elif opt1 == 'q':
    print('Bye!\n')
    sys.exit(0)
    
else:
    print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
    sys.exit(0)

################################################ Evaluation ############################################################
print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
print ('\n Confusion Matrix \n')  
print (confusion_matrix(tst_cat, predicted))  

pr=precision_score(tst_cat, predicted, average='macro') 
print ('\n Precision:'+str(pr)) 

rl=recall_score(tst_cat, predicted, average='macro') 
print ('\n Recall:'+str(rl))

fm=f1_score(tst_cat, predicted, average='macro') 
print ('\n Macro Averaged F1-Score:'+str(fm))

# Evaluation
print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
print ('\n Confusion Matrix \n')  
print (confusion_matrix(tst_cat, predicted))  

pr=precision_score(tst_cat, predicted, average='micro') 
print ('\n Precision:'+str(pr)) 

rl=recall_score(tst_cat, predicted, average='micro') 
print ('\n Recall:'+str(rl))

fm=f1_score(tst_cat, predicted, average='micro') 
print ('\n Micro Averaged F1-Score:'+str(fm))


# In[ ]:


# cv = CountVectorizer(max_features = 1420)


# In[ ]:


# X = cv.fit_transform(corpus).toarray()
# y = df.iloc[:, -1].values


# In[ ]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[ ]:


# X_train


# In[ ]:


# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)


# In[ ]:


# y_pred = classifier.predict(X_test)

# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# accuracy_score(y_test, y_pred)


# In[ ]:


# from sklearn.naive_bayes import MultinomialNB
# classifier = MultinomialNB()
# classifier.fit(X_train, y_train)


# In[ ]:


# y_pred = classifier.predict(X_test)

# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# accuracy_score(y_test, y_pred)


# In[ ]:


# classifier = RandomForestClassifier()
# classifier.fit(X_train, y_train)


# In[ ]:


# y_pred = classifier.predict(X_test)

# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# accuracy_score(y_test, y_pred)


# In[ ]:




