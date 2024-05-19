#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys

# In[2]:


try:
    if len(sys.argv) > 1:
        train_size = int(sys.argv[1])
        if train_size < 20 or train_size > 80:
            print("Train size argument must be between 20 and 80. Defaulting to 80.")
            train_size = 80
    else:
        print("Train size must argument is out of above 20 and above 80. Defaulting to 80.")
        train_size = 80
except ValueError:
    print("the number of arguments provided is Not one(none, two or more). Defaulting to 80.")
    train_size = 80


# In[3]:
print('Juhyun, Jung, A20521244 solution :')
print(f'Training set size : {train_size} %\n')

print('Training classifier...')

# 1. Load data 
data_file = open("yelp_academic_dataset_review.json")
data = []
total_length = 30000
c=0
with open("yelp_academic_dataset_review.json", encoding="utf-8") as data_file:
    for line in data_file:
        data.append(json.loads(line))
        c += 1
        if c == total_length:
            break
review_df = pd.DataFrame(data)
data_file.close()


# In[4]:


# 2.  Text pre-processing 


stopWordsCorpus = nltk.corpus.stopwords.words('english')
for char in string.punctuation:
    stopWordsCorpus.append(char)


def clean_and_split(text):
    tokenized = word_tokenize(text)
    #print(words)
    words = [w.lower() for w in tokenized if w.lower() not in stopWordsCorpus]
    return words

def lemmatize_words(word_list):
    return [lemmatizer.lemmatize(word) for word in word_list]

df = pd.DataFrame({'stars': review_df['stars'],'text':review_df['text']})
df['text'] = df['text'].apply(clean_and_split)

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
#print(df['text'][0])
df['text'] = df['text'].apply(lemmatize_words)

#print(df['text'][0])


# In[5]:


# 3. Train and Test data set

# setting train size
split_ratio =train_size*0.01
train_size_split = int(split_ratio*len(df))
test_size_split = int(0.2*len(df))

#print('train_size = ',train_size)

# split train and test data 
train_data_x = df['text'][:train_size_split]
test_data_x = df['text'][-test_size_split:]

train_data_y = df['stars'][:train_size_split]
test_data_y = df['stars'][-test_size_split:]

test_data_x_list = test_data_x.tolist()
test_data_y_list = test_data_y.tolist()

#print('train_data length: ',len(train_data_x))
#print('test_data length: ',len(test_data_x))

# frequency 
from collections import Counter

# Assuming df is your DataFrame with the 'text' column cleaned and split into words

# Create an empty Counter object to store word frequencies
word_freq_counter = Counter()

# Iterate over each row in the 'text' column and update the word frequency counter
for words_list in train_data_x:
    word_freq_counter.update(words_list)
Vocab_size = len(word_freq_counter)
#print('Vocab size : ',Vocab_size)


# In[6]:


# 4. make feature vectors

word_index = {word: idx for idx, word in enumerate(word_freq_counter)}
#word_index

def create_feature_vector(document):
    feature_vector = [0]*Vocab_size
    for word in document:
        if word in word_index:
            feature_vector[word_index[word]] += 1
    return feature_vector

#feature_vectors = {word:create_feature_vector(word,Vocab_size) for word in train_data[0]}
#feature_vectors
feature_vectors = train_data_x.apply(create_feature_vector)
#print(feature_vectors[:4])

# for checking 
#print(len(train_data_x[2]),'=?' ,sum(feature_vectors[2]))

# count each star

train_data_y=train_data_y.astype(np.int64)
label_total = {i:len(feature_vectors[train_data_y==i]) for i in range(1,6)}
#print(label_total)

# count
star_total_list=[]
for star in range(1,6):
    star_num = feature_vectors[train_data_y==star]
    star_num_array=np.array(star_num.tolist())
    sum_of_star_num = np.sum(star_num_array, axis=0)
    star_total_list.append(sum_of_star_num)





# In[7]:


# 5 calculate probability 
cond_prob_dic = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
alpha = 1
Vocab_list = list(word_index.keys())
# print(Vocab_list)
# print('-'*50)
for i in range(5):
    star = i+1
    j=0
    total_length = star_total_list[i].sum()
    # print(f'total_length of label {star} : ',total_length)
    for word in Vocab_list:
        index = word_index[word]
        # if j<2:
        #     print('word : ',word ,'/ up:',star_total_list[i][index]+1, 'down:',(total_length+alpha*Vocab_size))
        cond_prob_dic[star][word] = (star_total_list[i][index]+1)/(total_length+alpha*Vocab_size)
        # j+=1
    # print('-'*50)

# calculate P(y)
label_total

prob_y_dic ={}

total_train = sum(label_total.values())
for star in range(1,6):
    prob_y = label_total[star]/total_train
    prob_y_dic[star]=prob_y

# print(prob_y_dic)


# In[8]:


def classifier(filtered_S):
    logprob_list=[]
    for star in range(1,6):
        logprob = np.log(prob_y_dic[star])
        for word in filtered_S:
            #print(word)
            if word in word_index:
                logprob = logprob+np.log(cond_prob_dic[star][word])
        logprob_list.append(logprob)
    return logprob_list

def predict(x_data,y_data):
    """
    input : x_data, y_data
    output : logprob_list, x_data_list,y_data_list, y_predicted_list
    
    """
    x_data_list = x_data.tolist()
    y_data_list = y_data.tolist()
    y_predicted_list =[]
    logprob_list=[]
    total = len(x_data_list)
    for i in range(total):
        logprob = classifier(x_data_list[i])
        y_predicted = np.argmax(logprob)+1
        logprob_list.append(logprob)
        y_predicted_list.append(y_predicted)
    
    return logprob_list, x_data_list,y_data_list,y_predicted_list 
        
def accuracy_overall(x_data_list,y_data_list,y_predicted_list):
    total = len(x_data_list)
    # print('total : ',total)
    correct=0
    for i in range(total):
        #print('y_data_list :',y_data_list[i])
        #print('y_predicted_list :',y_predicted_list[i])
        if y_data_list[i] == y_predicted_list[i]:
            correct +=1
            #print('correct!')
    # print('correct :',correct)
    return correct/total
print('Testing classifier...')
print('Test results / metrics :','\n')
logprob_list, x_data_list,y_data_list,y_predicted_list = predict(test_data_x,test_data_y)


# In[9]:


def find_5_class_confusion_matrix(y,y_predicted):
    df = pd.DataFrame({'y': y_data_list,'y_predicted':y_predicted_list})
    confusion_matrix = pd.DataFrame(index=['Actual_1','Actual_2','Actual_3','Actual_4','Actual_5'],
                                    columns=['Predicted_1','Predicted_2','Predicted_3','Predicted_4','Predicted_5'])
    # i is Actual 
    # j is predicted 
    for i in range(5):
        for j in range(5):
            confusion_matrix.loc[f'Actual_{i+1}'][f'Predicted_{j+1}'] = len(df[(df['y']==(i+1)) & (df['y_predicted']==(j+1))])
    return confusion_matrix

def find_2_class_confusion_matrix(matrix,star):
    """
    input : 
            matrix : 5 class confusion matrix
            star : true label, star (int)
    
    output: 2 class_confusion matrix 
    
    """
    two_class_confusion_matrix = pd.DataFrame(index=['Actual_true','Actual_false'],
                                columns=['Predicted_true','Predicted_false'])
    
    two_class_confusion_matrix.loc['Actual_true','Predicted_true'] = matrix.loc[f'Actual_{star}',f'Predicted_{star}']
    two_class_confusion_matrix.loc['Actual_true','Predicted_false'] = matrix.loc[f'Actual_{star}',matrix.columns != f'Predicted_{star}'].sum()
    two_class_confusion_matrix.loc['Actual_false','Predicted_true'] = matrix.loc[matrix.index != f'Actual_{star}', f'Predicted_{star}'].sum()
    two_class_confusion_matrix.loc['Actual_false','Predicted_false'] = matrix.loc[matrix.index != f'Actual_{star}', matrix.columns != f'Predicted_{star}'].sum().sum()
    return two_class_confusion_matrix

def Sensitivity(matrix):
    """
    Sensitivity(Recall) = TP/(TP+FN)
    
    """
    up = matrix.loc['Actual_true']['Predicted_true']
    down = matrix.loc['Actual_true']['Predicted_true'] + matrix.loc['Actual_true']['Predicted_false']
    return up/down

def Specificity(matrix):
    """
    Specificity = TN/(TN+FP)
    
    """
    TN = matrix.loc['Actual_false']['Predicted_false']
    FP = matrix.loc['Actual_false']['Predicted_true']
    return (TN)/(TN+FP)

def Precision(matrix):
    """
    precision = TP/(TP+FP)
    
    """
    TP = matrix.loc['Actual_true']['Predicted_true']
    FP = matrix.loc['Actual_false']['Predicted_true']
    return TP/(TP+FP)

def Negative_predictive_value(matrix):
    """
    Negative predictive value = TN/(TN+FN)
    
    """
    TN = matrix.loc['Actual_false']['Predicted_false']
    FN = matrix.loc['Actual_true']['Predicted_false']
    return TN/(TN+FN)

def Accuracy(matrix):
    """
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    
    """
    TP = matrix.loc['Actual_true']['Predicted_true']
    TN = matrix.loc['Actual_false']['Predicted_false']
    FP = matrix.loc['Actual_false']['Predicted_true']
    FN = matrix.loc['Actual_true']['Predicted_false']
    return (TP+TN)/(TP+TN+FP+FN)

def F_score(matrix):
    """
    F-score  = TP/(TP + 0.5 * (FP+FN))
    
    """
    TP = matrix.loc['Actual_true']['Predicted_true']
    FP = matrix.loc['Actual_false']['Predicted_true']
    FN = matrix.loc['Actual_true']['Predicted_false']
    return TP/(TP + 0.5 * (FP+FN))

def microaverage_precision(matrix):
    """
    matrix : five class confusion matrix 
    microaverage precision = TP/(TP+FN)

    """
    star_1 = find_2_class_confusion_matrix(matrix,1)
    star_2 = find_2_class_confusion_matrix(matrix,2)
    star_3 = find_2_class_confusion_matrix(matrix,3)
    star_4 = find_2_class_confusion_matrix(matrix,4)
    star_5 = find_2_class_confusion_matrix(matrix,5)

    pooled_matrix = star_1+star_2+star_3+star_4+star_5

    TP = pooled_matrix.loc['Actual_true']['Predicted_true']
    FN = pooled_matrix.loc['Actual_true']['Predicted_false']

    return TP/(TP+FN)
    
def macroaverage_precision(matrix):
    """
    matrix : five class confusion matrix 

    macroaverage precision = average of precisions 
    """
    two_class_confusion_matrix_star_1 = find_2_class_confusion_matrix(matrix,1)
    two_class_confusion_matrix_star_2 = find_2_class_confusion_matrix(matrix,2)
    two_class_confusion_matrix_star_3 = find_2_class_confusion_matrix(matrix,3)
    two_class_confusion_matrix_star_4 = find_2_class_confusion_matrix(matrix,4)
    two_class_confusion_matrix_star_5 = find_2_class_confusion_matrix(matrix,5)

    one = Precision(two_class_confusion_matrix_star_1)
    two = Precision(two_class_confusion_matrix_star_2)
    three = Precision(two_class_confusion_matrix_star_3)
    four = Precision(two_class_confusion_matrix_star_4)
    five = Precision(two_class_confusion_matrix_star_5)
    
    return (one+two+three+four+five)/5 


def display(matrix,true):
    """
    matrix : two_class_confusion_matrix
    """
    TP = matrix.loc['Actual_true']['Predicted_true']
    TN = matrix.loc['Actual_false']['Predicted_false']
    FP = matrix.loc['Actual_false']['Predicted_true']
    FN = matrix.loc['Actual_true']['Predicted_false']
    print(f'Class {true}: star {true}')
    print("Number of true positives: ",TP)
    print("Number of true negatives: ",TN)
    print("Number of false positives: ",FP)
    print("Number of false negatives: ",FN)
    print("Sensitivity (recall): ", Sensitivity(matrix))
    print("Precision: ", Precision(matrix) )
    print("Negative predictive value: ",Negative_predictive_value(matrix))
    print("Accuracy: ",Accuracy(matrix))
    print("F_score: ",F_score(matrix))
    print('-'*50)


five_class_confusion_matrix = find_5_class_confusion_matrix(y_data_list,y_predicted_list)
two_class_confusion_matrix_star_1 = find_2_class_confusion_matrix(five_class_confusion_matrix,1)
two_class_confusion_matrix_star_2 = find_2_class_confusion_matrix(five_class_confusion_matrix,2)
two_class_confusion_matrix_star_3 = find_2_class_confusion_matrix(five_class_confusion_matrix,3)
two_class_confusion_matrix_star_4 = find_2_class_confusion_matrix(five_class_confusion_matrix,4)
two_class_confusion_matrix_star_5 = find_2_class_confusion_matrix(five_class_confusion_matrix,5)


display(two_class_confusion_matrix_star_1,1)
display(two_class_confusion_matrix_star_2,2)
display(two_class_confusion_matrix_star_3,3)
display(two_class_confusion_matrix_star_4,4)
display(two_class_confusion_matrix_star_5,5)

print('overall accuracy : ',accuracy_overall(x_data_list,y_data_list,y_predicted_list))
print('Microaverage precision : ',microaverage_precision(five_class_confusion_matrix))
print('Macroaverage precision : ',macroaverage_precision(five_class_confusion_matrix))


# In[12]:


while True:
    S = str(input("Enter your setence : \n"))

    print("Sentence S: \n\n",S)

    
    preprocessed_S = lemmatize_words(clean_and_split(S))
    logprob_list = classifier(preprocessed_S)
    predicted_y = np.argmax(logprob_list)+1
    print('was classified as ',predicted_y,'.')
    for i in range(5):
        print(f'P ( star {i+1} | S ) =', np.exp(logprob_list[i]))

    answer = str(input("Do you want to enter another sentence [Y/N] ? "))

    if answer == 'Y':
        print('-'*50)
    elif answer == 'N':
        print("END")
        break
    else:
        print("Invalid answer")
        print("END")
        break





# In[ ]:




