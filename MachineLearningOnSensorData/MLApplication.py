#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
import os
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import pywt
import scipy as sp
import scipy.fftpack
from scipy.fftpack import fft
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.plotting.register_matplotlib_converters()
import seaborn as sns
from sklearn import *
import tensorflow as tf


# In[2]:


# Variables
EMG_headers = pd.Series(['EMG 1', 'EMG 2', 'EMG 3', 'EMG 4', 'EMG 5', 'EMG 6', 'EMG 7', 'EMG 8'])
dict_all_users = {}
fft_dict = {}


# In[3]:


def eat_noeat_determine(raw_emg_df, directory, df_file_name):
    for file_name in os.listdir(directory):
        if not 'EMG' in file_name:
            if file_name.split('.')[0] == df_file_name.split('_')[0]:
                temp_gt_path = os.path.join(directory, file_name)
                temp_gt_df = pd.read_csv(temp_gt_path, names = ['Start', 'Stop', 'Discard'], header = None)
                temp_gt_df = (temp_gt_df * (100/30)).round(0).astype(int)
                for emg_i in raw_emg_df.index:
                    for gt_i in temp_gt_df.index:
                        if (emg_i >= temp_gt_df.loc[gt_i, 'Start']) & (emg_i <= temp_gt_df.loc[gt_i, 'Stop']):
                            raw_emg_df.loc[emg_i, 'Eat'] = True
                return raw_emg_df


# In[4]:


# Apologies, but I have timed this step to take about 20 minutes on my computer
def scan_folder(parent):
    for file_name in os.listdir(parent):
        if 'EMG' in file_name:
            temp_user = os.path.basename(parent)
            temp_emg_path = os.path.join(parent, file_name)
            temp_emg_df = pd.read_csv(temp_emg_path, names = ['Unix', 'EMG 1', 'EMG 2', 'EMG 3', 'EMG 4', 'EMG 5', 'EMG 6', 'EMG 7', 'EMG 8'], header = None)
            temp_emg_df['Eat'] = False
            dict_all_users[file_name] = eat_noeat_determine(temp_emg_df, parent, file_name)
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                scan_folder(current_path)

# Insert the path where the files are store below.  Note to keep the r in the begining 
scan_folder(r"\UserFiles")

# In[5]:


raw_users_dict = dict_all_users
# End of importing block, above is used in both phases


# In[6]:


#PHASE 1 - CREATE DICTIONARY FOR KEEPING USERS SEPERATE
all_users = {}
k = 0
for key,val in raw_users_dict.items():
    temp_eating = raw_users_dict[key][raw_users_dict[key]['Eat']==True]
    temp_not_eating = raw_users_dict[key][raw_users_dict[key]['Eat']==False]
    # below makes not eating rows = eating rows
    temp_not_eating = temp_not_eating.iloc[0:len(temp_eating.index)] 
    all_users[k] = pd.concat([temp_eating, temp_not_eating])
    all_users[k] = all_users[k].reset_index(drop=True) 
    k += 1
    
combined_dict = {}
j = 8
i = 0
while i < len(all_users.keys()):
    i += 2
    j += 1
    if j == 15 or j == 20 or j == 35:
        i -= 2
        continue
    str_j = str(j)
    key_name = 'user_' + str_j
    combined_dict[key_name] = pd.concat([all_users[i-2], all_users[i-1]])
    combined_dict[key_name] = combined_dict[key_name].reset_index(drop=True)


# In[7]:


# Phase 1 objective: split each user according to 60/40
p1_raw_split = {}
def p1_test_train_split(dictionary):
    temp_dict = {}
    p1_train_df = pd.DataFrame()
    p1_test_df = pd.DataFrame()
    for key, val in dictionary.items():
        # establish length to extract train test
        p1_train_len = int(len(dictionary[key][dictionary[key]['Eat']==True].index)*0.6)
        # p1 train
        p1_eat_train = dictionary[key][dictionary[key]['Eat']==True].iloc[0:p1_train_len]
        p1_train_df = p1_train_df.append(p1_eat_train)
        p1_noeat_train = dictionary[key][dictionary[key]['Eat']==False].iloc[0:p1_train_len]
        p1_train_df = p1_train_df.append(p1_noeat_train)      
        # p1 test
        p1_eat_test = dictionary[key][dictionary[key]['Eat']==True].iloc[p1_train_len:-1]
        p1_test_df = p1_test_df.append(p1_eat_test)
        p1_noeat_test = dictionary[key][dictionary[key]['Eat']==False].iloc[p1_train_len:-1]
        p1_test_df = p1_test_df.append(p1_noeat_test)    
    
    p1_train_df = p1_train_df.reset_index(drop=True)
    temp_dict['p1_train'] = p1_train_df
    
    p1_test_df = p1_test_df.reset_index(drop=True)
    temp_dict['p1_test'] = p1_test_df  
    
    return temp_dict
    
p1_raw_split = p1_test_train_split(combined_dict)


# In[8]:


# phase 2 objective: split all users 60/40 i.e. train = user 9-user 27; test = user 28-user 41
# Note: the 60/40 user split above was done so that training contained 60% of data and testing contained 40%
p2_raw_split = {}
def p2_test_train_split(dictionary):
    temp_dict = {}
    p2_train_df = pd.DataFrame()
    p2_test_df = pd.DataFrame()
    i = 0
    for key, val in dictionary.items():
        user_number = int(key.split('_')[1])
        if user_number < 28:
            p2_train_df = p2_train_df.append(dictionary[key])
        else:
            p2_test_df = p2_test_df.append(dictionary[key])
    
    p2_train_df = p2_train_df.reset_index(drop=True)
    temp_dict['p2_train'] = p2_train_df
    
    p2_test_df = p2_test_df.reset_index(drop=True)
    temp_dict['p2_test'] = p2_test_df 
       
    return temp_dict
            
p2_raw_split = p2_test_train_split(combined_dict)

p1_p2_raw = {**p1_raw_split, **p2_raw_split}


# In[10]:


# Phase 1 & 2 Split Outputs
p1_p2_raw['p1_train'].to_csv('p1 training.csv')
p1_p2_raw['p1_test'].to_csv('p1 testing.csv')
p1_p2_raw['p2_train'].to_csv('p2 training.csv')
p1_p2_raw['p2_test'].to_csv('p2 testing.csv')


# In[11]:


# 20 refers to how many rows to average over
def scale_with_average(dictionary, agg_step):
    temp_scale_dict = dictionary
    for key, val in temp_scale_dict.items():
        temp_scale_dict[key] = temp_scale_dict[key].drop(['Unix'], axis=1)
        temp_eating = temp_scale_dict[key][temp_scale_dict[key]['Eat']==True]
        temp_eating = temp_eating.groupby(np.arange(len(temp_eating))//agg_step).mean()        
        temp_not_eating = temp_scale_dict[key][temp_scale_dict[key]['Eat']==False]
        temp_not_eating = temp_not_eating.groupby(np.arange(len(temp_not_eating))//agg_step).mean()
        temp_scale_dict[key] = pd.concat([temp_eating, temp_not_eating])
        temp_scale_dict[key] = temp_scale_dict[key].reset_index(drop=True)
    return temp_scale_dict

scaled_dict = scale_with_average(p1_p2_raw, 300)


# In[12]:


fe_dict = {}
def features(dictionary): 
    feature_dict = dictionary
    for key, val in feature_dict.items():
        feature_dict[key]['Max'] = feature_dict[key][EMG_headers].max(axis=1)
        feature_dict[key]['RMS'] = np.sqrt((feature_dict[key]['EMG 1']**2 + feature_dict[key]['EMG 2']**2 + feature_dict[key]['EMG 3']**2 + feature_dict[key]['EMG 4']**2 + feature_dict[key]['EMG 5']**2 + feature_dict[key]['EMG 6']**2 + feature_dict[key]['EMG 7']**2 + feature_dict[key]['EMG 8']**2)/8)
        feature_dict[key]['Mean'] = feature_dict[key][EMG_headers].mean(axis=1)
        feature_dict[key]['StD'] = feature_dict[key][EMG_headers].std(axis=1)
        feature_dict[key]['Min'] = feature_dict[key][EMG_headers].min(axis=1)
        # calculate fft of each row
        temp_df = feature_dict[key].drop(['Eat', 'Max', 'RMS', 'Mean', 'StD', 'Min'], axis=1)
        fft_list = []
        for i in temp_df.index:
            x = temp_df.iloc[i].values
            y = fft(x)
            fft_list.append(y[0].real)
        fft_dict[key] = pd.DataFrame(fft_list, columns=['FFT'])

    for key,val in feature_dict.items():
        if not 'FFT' in feature_dict[key].columns:
            for key1, val in feature_dict.items():
                for key2, val in fft_dict.items():
                    if key1 == key2:
                        feature_dict[key1] = pd.concat([feature_dict[key1], fft_dict[key2]], axis=1)
    return feature_dict

fe_dict = features(scaled_dict)


# In[13]:


shuffle_dict = {}
for key, val in fe_dict.items():
    shuffle_dict[key] = fe_dict[key].sample(frac=1).reset_index(drop=True)   


# In[14]:


pca_dict = {}
target_dict = {}
scaled_pca = {}
pca_components = {}
transform_pca = {}
eigenvectors = {}
eigenvector_matrix = {}

for key, val in shuffle_dict.items():
    target_dict[key] = pd.DataFrame(shuffle_dict[key]['Eat'].astype(int), columns=['Eat'])
    pca_dict[key] = shuffle_dict[key].drop(['EMG 1', 'EMG 2', 'EMG 3', 'EMG 4', 'EMG 5', 'EMG 6', 'EMG 7', 'EMG 8', 'Eat'], axis=1)
    scaled_pca[key] = StandardScaler().fit_transform(pca_dict[key])
    pca_components[key] = PCA(n_components = 3)
    transform_pca[key] = pca_components[key].fit_transform(scaled_pca[key])
    eigenvectors[key] = pca_components[key].components_ # eigenvectors of top 2 PC's based on index 2
    eigenvector_matrix[key] = np.hstack((eigenvectors[key][0,:].reshape(6,1), eigenvectors[key][1,:].reshape(6,1), 
                                        eigenvectors[key][2,:].reshape(6,1)))


# In[15]:


# Phase 1 & 2 Projected Feature Matrix
projected_features_dict = {}   
    
for key, val in pca_dict.items():
    projected_features_dict[key] = pca_dict[key].dot(eigenvector_matrix[key])
    projected_features_dict[key].rename(columns = {0:'Feature 1', 1:'Feature 2', 2:'Feature 3'}, inplace = True)
    projected_features_dict[key] = pd.concat([projected_features_dict[key], target_dict[key]], axis=1)


# In[16]:


# Phase 1 & 2 machine learning inputs
X = {'p1_X_train': projected_features_dict['p1_train'][['Feature 1', 'Feature 2', 'Feature 3']],                                                    
     'p1_X_test': projected_features_dict['p1_test'][['Feature 1', 'Feature 2', 'Feature 3']], 
     'p2_X_train': projected_features_dict['p2_train'][['Feature 1', 'Feature 2', 'Feature 3']], 
     'p2_X_test': projected_features_dict['p2_test'][['Feature 1', 'Feature 2', 'Feature 3']]
    }
y = {'p1_y_train': projected_features_dict['p1_train'][['Eat']], 
     'p1_y_test': projected_features_dict['p1_test'][['Eat']], 
     'p2_y_train': projected_features_dict['p2_train'][['Eat']], 
     'p2_y_test': projected_features_dict['p2_test'][['Eat']]
    }


# In[17]:


# Phase 1 Decision Tree Model
dtree = DecisionTreeClassifier(class_weight={0:50})

p1_dt_model = dtree.fit(X['p1_X_train'], y['p1_y_train'])

p1_dt_proba = pd.DataFrame(p1_dt_model.predict_proba(X['p1_X_test'])[:,1], columns=['proba'])
threshold = 0.8 # You can play on this value (default is 0.5)
p1_dt_proba['predict'] = p1_dt_proba['proba'].apply(lambda el: 1.0 if el >= threshold else 0.0)

# Phase 1 Decision Tree Outputs
p1_dt_confuse = pd.DataFrame(confusion_matrix(y['p1_y_test'], p1_dt_proba['predict']), index=['Actual Not Eating', 'Actual Eating'], columns=['Predict Not Eating', 'Predict Eating'])
p1_dt_class = pd.DataFrame(classification_report(y['p1_y_test'], p1_dt_proba['predict'], output_dict=True)).transpose()
p1_dt_accuracy = pd.DataFrame(accuracy_score(y['p1_y_test'], p1_dt_proba['predict']),index=['Phase 1'], columns=['Accuracy'])

p1_dt_confuse.to_csv('p1 dt confusion matrix.csv')
p1_dt_class.to_csv('p1 dt classification report.csv')
p1_dt_accuracy.to_csv('p1 dt accuracy.csv')


# In[18]:


# Phase 2 Decision Tree Model
p2_dt_model = dtree.fit(X['p2_X_train'], y['p2_y_train'])
p2_dt_predict = p2_dt_model.predict(X['p2_X_test'])

# Phase 2 DT outputs
p2_dt_confuse = pd.DataFrame(confusion_matrix(y['p2_y_test'], p2_dt_predict), index=['Actual Not Eating', 'Actual Eating'], columns=['Predict Not Eating', 'Predict Eating'])
p2_dt_class = pd.DataFrame(classification_report(y['p2_y_test'], p2_dt_predict, output_dict=True)).transpose()
p2_dt_accuracy = pd.DataFrame(accuracy_score(y['p2_y_test'], p2_dt_predict),index=['Phase 2'], columns=['Accuracy'])

p2_dt_confuse.to_csv('p2 dt confusion matrix.csv')
p2_dt_class.to_csv('p2 dt classification report.csv')
p2_dt_accuracy.to_csv('p2 dt accuracy.csv')


# In[19]:


# Phase 1 Support Vector Machine Model
svm_model = SVC(C=10, gamma=1, kernel='rbf', cache_size=1000)

p1_svm_model = svm_model.fit(X['p1_X_train'], y['p1_y_train'].values.ravel())
p1_svm_predict = p1_svm_model.predict(X['p1_X_test'])

# Phase 1 Support Vector Machine Outputs
p1_svm_confuse = pd.DataFrame(confusion_matrix(y['p1_y_test'].values.ravel(), p1_svm_predict), 
                              index=['Actual Not Eating', 'Actual Eating'], columns=['Predict Not Eating', 'Predict Eating'])
p1_svm_class = pd.DataFrame(classification_report(y['p1_y_test'].values.ravel(), p1_svm_predict, output_dict=True)).transpose()
p1_svm_accuracy = pd.DataFrame(accuracy_score(y['p1_y_test'].values.ravel(), p1_svm_predict), 
                               index=['Phase 1'], columns=['Accuracy'])

p1_svm_confuse.to_csv('p1 svm confusion matrix.csv')
p1_svm_class.to_csv('p1 svm classification report.csv')
p1_svm_accuracy.to_csv('p1 svm accuracy.csv')


# In[20]:


#Phase 2 SVM Model
p2_svm_model = svm_model.fit(X['p2_X_train'], y['p2_y_train'].values.ravel())
p2_svm_predict = p2_svm_model.predict(X['p2_X_test'])

# Phase 2 Support Vector Machine Outputs
p2_svm_confuse = pd.DataFrame(confusion_matrix(y['p2_y_test'].values.ravel(), p2_svm_predict), 
                              index=['Actual Not Eating', 'Actual Eating'], columns=['Predict Not Eating', 'Predict Eating'])
p2_svm_class = pd.DataFrame(classification_report(y['p2_y_test'].values.ravel(), p2_svm_predict, output_dict=True)).transpose()
p2_svm_accuracy = pd.DataFrame(accuracy_score(y['p2_y_test'].values.ravel(), p2_svm_predict),index=['Phase 2'], columns=['Accuracy'])

p2_svm_confuse.to_csv('p2 svm confusion matrix.csv')
p2_svm_class.to_csv('p2 svm classification report.csv')
p2_svm_accuracy.to_csv('p2 svm accuracy.csv')


# In[21]:


#NN model for Phase 1 & 2
nn_model = Sequential()
nn_model.add(Dense(6, activation='relu', input_dim=3, kernel_regularizer=regularizers.l2(0.0001)))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'], n_jobs=-1)

early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience = 20)


# In[22]:


# Phase 1 Neural Network Inputs
scaler =  MinMaxScaler()
p1_nn_X_train = scaler.fit_transform(X['p1_X_train'].values)
p1_nn_X_test = scaler.transform(X['p1_X_test'].values)

# Phase 1 Model
nn_model.fit(x=p1_nn_X_train, y=y['p1_y_train'].values, epochs=300, batch_size=50, callbacks=[early_stop])

# predictions from x test
p1_nn_predict = nn_model.predict_classes(p1_nn_X_test)

# Phase 1 NN Outputs
p1_nn_confuse = pd.DataFrame(confusion_matrix(y['p1_y_test'].values.ravel(), p1_nn_predict), 
                              index=['Actual Not Eating', 'Actual Eating'], columns=['Predict Not Eating', 'Predict Eating'])
p1_nn_class = pd.DataFrame(classification_report(y['p1_y_test'].values.ravel(), p1_nn_predict, output_dict=True)).transpose()
p1_nn_accuracy = pd.DataFrame(accuracy_score(y['p1_y_test'].values.ravel(), p1_nn_predict), 
                               index=['Phase 1'], columns=['Accuracy'])

p1_nn_confuse.to_csv('p1 nn confusion matrix.csv')
p1_nn_class.to_csv('p1 nn classification report.csv')
p1_nn_accuracy.to_csv('p1 nn accuracy.csv')


# In[23]:


# Phase 1 Neural Network Inputs
scaler =  MinMaxScaler()
p2_nn_X_train = scaler.fit_transform(X['p2_X_train'].values)
p2_nn_X_test = scaler.transform(X['p2_X_test'].values)

# Phase 1 Model
nn_model.fit(x=p2_nn_X_train, y=y['p2_y_train'].values, epochs=300, batch_size=50, callbacks=[early_stop])

# predictions from x test
p2_nn_predict = nn_model.predict_classes(p2_nn_X_test)

# Phase 1 NN Outputs
p2_nn_confuse = pd.DataFrame(confusion_matrix(y['p2_y_test'].values.ravel(), p2_nn_predict), 
                              index=['Actual Not Eating', 'Actual Eating'], columns=['Predict Not Eating', 'Predict Eating'])
p2_nn_class = pd.DataFrame(classification_report(y['p2_y_test'].values.ravel(), p2_nn_predict, output_dict=True)).transpose()
p2_nn_accuracy = pd.DataFrame(accuracy_score(y['p2_y_test'].values.ravel(), p2_nn_predict), 
                               index=['Phase 1'], columns=['Accuracy'])

p2_nn_confuse.to_csv('p2 nn confusion matrix.csv')
p2_nn_class.to_csv('p2 nn classification report.csv')
p2_nn_accuracy.to_csv('p2 nn accuracy.csv')

