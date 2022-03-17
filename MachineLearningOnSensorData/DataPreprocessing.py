#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.fftpack import fft
import glob
import pywt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.plotting.register_matplotlib_converters()
from sklearn import *


# In[2]:


# This was my first CS project and likely could have been implemented in a more efficient manner


# In[3]:


# Variables

EMG_headers = pd.Series(['EMG 1', 'EMG 2', 'EMG 3', 'EMG 4', 'EMG 5', 'EMG 6', 'EMG 7', 'EMG 8'])
dict_all = {} # contains raw data assigned to keys representing file names
fft_dict = {} # contains fft results calucated row-wise
dict_final = {} # contains feature extraction data for plotting & PCA
PCA_df = pd.DataFrame() # dataframe representing feature matrix for Phase 3 Subtask 1 and used in PCA


# In[4]:


# methods

def normalize(df): # normalizes raw data set, but removed this from the project
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(x_scaled, columns = EMG_headers)
    return df_normalized

def average_rows(dictionary_set, avg_step): # scale
    for key, val in dictionary_set.items():
        if 'eat' in key:
            dict_final[key] = dictionary_set[key].drop(EMG_headers, axis=1)
            dict_final[key] = dict_final[key].groupby(np.arange(len(dict_final[key]))//avg_step).mean()
            dict_final[key] = dict_final[key].reset_index(drop=True)
            dict_final[key].loc[:,'Time (ms)'] = (dict_final[key].index)
        if 'spoon_no' in key:
            dict_final[key] = dict_final[key].iloc[0:len(dict_final['spoon_eat'].index)]
            dict_final[key] = dict_final[key].reset_index(drop=True)
            dict_final[key].loc[:,'Time (ms)'] = (dict_final[key].index)
        if 'fork_no' in key:
            no_avg_step = len(dict_final[key].index) / len(dict_final['fork_eat'].index)
            dict_final[key] = dict_final[key].iloc[0:len(dict_final['fork_eat'].index)]
            dict_final[key] = dict_final[key].reset_index(drop=True)
            dict_final[key].loc[:,'Time (ms)'] = (dict_final[key].index)

def phaseI_eat_out(dict_raw):
    eating = pd.concat([dict_raw['spoon_eat'], dict_raw['fork_eat']])
    return eating.reset_index(drop=True)

def phaseI_noeat_out(dict_raw):
    not_eating = pd.concat([dict_raw['spoon_no_eat'], dict_raw['fork_no_eat']])
    return not_eating.reset_index(drop=True)


# In[5]:


#Phase 1
def Phase_I():
    for filename in glob.glob("*.txt"):
        if "EMG" in filename:
            dict_all[filename[-10:-4]] = pd.read_csv(filename, names = ['Unix', 'EMG 1', 'EMG 2', 'EMG 3', 'EMG 4', 'EMG 5', 'EMG 6', 'EMG 7', 'EMG 8'], header = None)
            dict_all[filename[-10:-4]] = dict_all[filename[-10:-4]].drop(['Unix'], axis=1)
            #dict_all[filename[-10:-4]] = normalize(dict_all[filename[-10:-4]])
            dict_all[filename[-10:-4]]['Eat'] = False
        else:
            dict_all[filename[-6:-4]] = (((pd.read_csv(filename, names = ['Start', 'Stop', 'Discard'], header = None))*100)/30).round(0).astype(int)

    for EMG_spoon_i in dict_all['79_EMG'].index:
        for GT_spoon_i in dict_all['79'].index:
            if (EMG_spoon_i >= dict_all['79'].loc[GT_spoon_i, 'Start']) & (EMG_spoon_i <= dict_all['79'].loc[GT_spoon_i, 'Stop']):
                dict_all['79_EMG'].loc[EMG_spoon_i, 'Eat'] = True

    for EMG_fork_i in dict_all['23_EMG'].index:
        for GT_fork_i in dict_all['23'].index:
            if (EMG_fork_i >= dict_all['23'].loc[GT_fork_i, 'Start']) & (EMG_fork_i <= dict_all['23'].loc[GT_fork_i, 'Stop']):
                dict_all['23_EMG'].loc[EMG_fork_i, 'Eat'] = True

    dict_all['spoon_eat'] = dict_all['79_EMG'][dict_all['79_EMG']['Eat']==True]
    dict_all['spoon_no_eat'] = dict_all['79_EMG'][dict_all['79_EMG']['Eat']==False]
    dict_all['fork_eat'] = dict_all['23_EMG'][dict_all['23_EMG']['Eat']==True]
    dict_all['fork_no_eat'] = dict_all['23_EMG'][dict_all['23_EMG']['Eat']==False]

Phase_I()
phaseI_eat_out(dict_all).to_csv('Phase I Eating Output.csv')
phaseI_noeat_out(dict_all).to_csv('Phase I Not Eating Output.csv')
#eating_matrix = phaseI_eat_out(dict_all) # run this to view Phase I answer
#not_eating_matrix = phaseI_noeat_out(dict_all) # run this to view Phase I answer


# In[6]:


# Phase 2

def Phase_II_Features(): # I believe i fixed it, but in previous versions running this more than once causes fft to continue to add columns
    for key, val in dict_all.items():
        if "eat" in key:
            dict_all[key].loc[:,'Max'] = dict_all[key][EMG_headers].max(axis=1)
            dict_all[key].loc[:,'Min'] = dict_all[key][EMG_headers].min(axis=1)
            dict_all[key].loc[:,'Mean'] = dict_all[key][EMG_headers].mean(axis=1)
            dict_all[key].loc[:,'StD'] = dict_all[key][EMG_headers].std(axis=1)
            dict_all[key].loc[:,'RMS'] = np.sqrt((dict_all[key]['EMG 1']**2 + dict_all[key]['EMG 2']**2 + dict_all[key]['EMG 3']**2 + dict_all[key]['EMG 4']**2 + dict_all[key]['EMG 5']**2 + dict_all[key]['EMG 6']**2 + dict_all[key]['EMG 7']**2 + dict_all[key]['EMG 8']**2)/8)
            # calculate fft of each row
            temp_df = dict_all[key].drop(['Eat', 'Max', 'Min', 'Mean', 'StD', 'RMS'], axis=1) #, 'Max', 'Min', 'Mean', 'StD', 'RMS'
            temp_df = temp_df.reset_index(drop=True)
            fft_list = []
            for i in temp_df.index:
                x = temp_df.iloc[i].values
                y = fft(x)
                fft_list.append(y[0].real)
            fft_dict[key] = pd.DataFrame(fft_list, columns=['FFT'])

    if not 'FFT' in dict_all['spoon_eat']:
        for key1, val in dict_all.items():
            for key2, val in fft_dict.items():
                if key1 == key2:
                    temp_index = dict_all[key1].index.astype(int)
                    fft_dict[key2] = fft_dict[key2].set_index(temp_index)
                    dict_all[key1] = pd.concat([dict_all[key1], fft_dict[key2]], axis=1)

        average_rows(dict_all, 4) # scales down rows by averaging over 3 rows at a time and creates the matrix dict_final which contains 4 matrices

Phase_II_Features()
dict_final.keys() # Phase_II_Features() produces this dictionary which contains 4 dataframes that hold feature extraction results in keys ['spoon_eat'],['spoon_no_eat'], [fork_eat'], ['fork_no_eat']
dict_all['spoon_eat'].to_csv('Phase II FE Spoon Eating Output.csv')
dict_all['spoon_no_eat'].to_csv('Phase II FE Spoon Not Eating Output.csv')
dict_all['fork_eat'].to_csv('Phase II FE Fork Eating Output.csv')
dict_all['fork_no_eat'].to_csv('Phase II FE Fork Not Eating Output.csv')


# In[7]:


# Phase 2 (d)

fe_figs, fe_axes = plt.subplots(nrows = 12, ncols = 1, figsize = (10,40), dpi = 200)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.3, hspace=0.3)

def plotting(dictionary_set):
    for i in range(12):
        if i < 5:
            fe_axes[i].plot(dictionary_set['spoon_eat']['Time (ms)'], dictionary_set['spoon_eat'].iloc[:,i+1], linewidth=.5, label='Spoon Eating')
            fe_axes[i].plot(dictionary_set['spoon_no_eat']['Time (ms)'], dictionary_set['spoon_no_eat'].iloc[:,i+1], linewidth=.5, label='Spoon Not Eating')
            fe_axes[i].set_title(dictionary_set['spoon_no_eat'].columns[i+1])
            fe_axes[i].set_xlabel('Time (ms)')
            fe_axes[i].set_ylabel('EMG')
            fe_axes[i].legend(loc='best')
        if i == 5:
            fe_axes[i].plot(dictionary_set['spoon_eat']['Time (ms)'], dictionary_set['spoon_eat'].iloc[:,i+1], linewidth=.5, label='Spoon Eating')
            fe_axes[i].plot(dictionary_set['spoon_no_eat']['Time (ms)'], dictionary_set['spoon_no_eat'].iloc[:,i+1], linewidth=.5, label='Spoon Not Eating')
            fe_axes[i].set_title(dictionary_set['spoon_no_eat'].columns[i+1])
            fe_axes[i].set_xlabel('Time (ms)')
            fe_axes[i].set_ylabel('EMG')
            fe_axes[i].legend(loc='best')
        if i > 5:
            fe_axes[i].plot(dictionary_set['fork_eat']['Time (ms)'], dictionary_set['fork_eat'].iloc[:,i-5], linewidth=.5, label='Fork Eating')
            fe_axes[i].plot(dictionary_set['fork_no_eat']['Time (ms)'], dictionary_set['fork_no_eat'].iloc[:,i-5], linewidth=.5, label='Fork Not Eating')
            fe_axes[i].set_title(dictionary_set['fork_no_eat'].columns[i-5])
            fe_axes[i].set_xlabel('Time (ms)')
            fe_axes[i].set_ylabel('EMG')
            fe_axes[i].legend(loc='best')
        if i == 12:
            fe_axes[i].plot(dictionary_set['fork_eat']['Time (ms)'], dictionary_set['fork_eat'].iloc[:,i-6], linewidth=.5, label='Fork Eating')
            fe_axes[i].plot(dictionary_set['fork_no_eat']['Time (ms)'], dictionary_set['fork_no_eat'].iloc[:,i-6], linewidth=.5, label='Fork Not Eating')
            fe_axes[i].set_title(dictionary_set['fork_no_eat'].columns[i-6])
            fe_axes[i].set_xlabel('Time (ms)')
            fe_axes[i].set_ylabel('EMG')
            fe_axes[i].legend(loc='best')

plotting(dict_final) # cannot save subplots so this will not download


# In[8]:


# Phase 3 Subtask 1
for key, val in dict_final.items():
    PCA_df = PCA_df.append([dict_final[key]])
target = PCA_df['Eat'].astype(int).values
PCA_df = PCA_df.drop(['Eat', 'Time (ms)'], axis=1)
PCA_df = PCA_df.reset_index(drop=True) # Phase 3 Subtask 1 Output, Note: it is not 1 matrix with 6 feature col and rows that include eating and non-eating data
phaseIII_task1_df = PCA_df # answer to Phase III subtask 1
phaseIII_task1_df.to_csv('Phase III ST1 Feature Matrix.csv')


# In[9]:


# Phase 3 Subtask 2
# transform to standard scale
scaler = StandardScaler() # each feature has single unit variance so PCA can be used
scaler.fit(PCA_df) # fit scaler to data frame
scaled_data = scaler.transform(PCA_df) # transform scaled data / has original 6 columns

#PCA
pca_2comps = PCA(n_components=2)
pca_2comps.fit(scaled_data)
transform_pca = pca_2comps.transform(scaled_data) # transform into principal components / causes x_pca to be 2 columns which is direction of 2 PCA's


# In[10]:


# Phase 3 Subtask 2
pca_figure = plt.figure(figsize=(8,6))
plt.scatter(transform_pca[:,0], transform_pca[:,1], c=target, s=8)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('User 32 Principal Components')
pca_figure.savefig('User 32 Principal Components')


# In[11]:


# compute eigenvalues and vectors of all 6 features
pca_6comps = PCA()
pca_6comps.fit(scaled_data)
eigenvectors_6 = pca_6comps.components_ # eigenvectors of all 6 features
eigenvalues_6 = pca_6comps.explained_variance_ # eigenvalues of all 6 features

e_val_6_total = sum(eigenvalues_6)
var_explained_6 = [(i / e_val_6_total) for i in sorted(eigenvalues_6, reverse=True)]
cumulative_var_explained_6 = np.cumsum(var_explained_6)

cumulative_var_figure = plt.figure(figsize=(8,6))
plt.bar(range(1,7), var_explained_6, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,7), cumulative_var_explained_6, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()
cumulative_var_figure.savefig('PCA Cumulative Variance')


# In[12]:


eigenvalue_labels = ['Eval1', 'Eval2', 'Eval3', 'Eval4', 'Eval5', 'Eval6']
eigenvec_labels = ['Evec1', 'Evec2', 'Evec3', 'Evec4', 'Evec5', 'Evec6']
ev_colors = ['b','g','r','c','m','y']

num_evs = len(eigenvec_labels)
angles = np.linspace(0, 2 * np.pi, num_evs, endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True)) # cannot save subplots so this will not download

def add_to_radar(eigenvector, color, labelev):
    values = eigenvector.tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=1, label=labelev)
    ax.fill(angles, values, color=color, alpha=0.25)

add_to_radar(eigenvectors_6[0], ev_colors[0], eigenvec_labels[0])
add_to_radar(eigenvectors_6[1], ev_colors[1], eigenvec_labels[1])
add_to_radar(eigenvectors_6[2], ev_colors[2], eigenvec_labels[2])
add_to_radar(eigenvectors_6[3], ev_colors[3], eigenvec_labels[3])
add_to_radar(eigenvectors_6[4], ev_colors[4], eigenvec_labels[4])
add_to_radar(eigenvectors_6[5], ev_colors[5], eigenvec_labels[5])

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles), eigenvalue_labels)

for label, angle in zip(ax.get_xticklabels(), angles):
  if angle in (0, np.pi):
    label.set_horizontalalignment('center')
  elif 0 < angle < np.pi:
    label.set_horizontalalignment('left')
  else:
    label.set_horizontalalignment('right')

ax.set_rlabel_position(180 / num_evs)

ax.tick_params(colors='#222222')
ax.tick_params(axis='y', labelsize=8)
ax.grid(color='#AAAAAA')
ax.spines['polar'].set_color('#222222')
ax.set_facecolor('#FAFAFA')

ax.set_title('Eigenvectors', y=1.08)

ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))


# In[13]:


# projection of 6 PCA's on each row of PCA_df (ie original)
projected_df = pd.DataFrame()
for i in PCA_df.index:
    projected_df.loc[i,'Projected Max'] = (PCA_df.iloc[i].values * eigenvectors_6[0].reshape(len(eigenvectors_6[0]),1)).sum()
    projected_df.loc[i,'Projected Min'] = (PCA_df.iloc[i].values * eigenvectors_6[1].reshape(len(eigenvectors_6[1]),1)).sum()
    projected_df.loc[i,'Projected Mean'] = (PCA_df.iloc[i].values * eigenvectors_6[2].reshape(len(eigenvectors_6[2]),1)).sum()
    projected_df.loc[i,'Projected StD'] = (PCA_df.iloc[i].values * eigenvectors_6[3].reshape(len(eigenvectors_6[3]),1)).sum()
    projected_df.loc[i,'Projected RMS'] = (PCA_df.iloc[i].values * eigenvectors_6[4].reshape(len(eigenvectors_6[4]),1)).sum()
    projected_df.loc[i,'Projected FFT'] = (PCA_df.iloc[i].values * eigenvectors_6[5].reshape(len(eigenvectors_6[5]),1)).sum()

projected_df.to_csv('Phase III ST4 Projection of FE.csv')


# In[ ]:
