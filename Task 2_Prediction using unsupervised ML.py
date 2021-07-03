#!/usr/bin/env python
# coding: utf-8

# # The Spark Fondation #GRIPJULY21
# ## Task 2:Prediction using unsupervised ML
# ### Author: Vamsi Vasamsetti

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


df = pd.read_csv('C:\\Users\\Vamsi\\Downloads\\Iris.csv')


# ### Exploratory Data Analysis

# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.describe(include='all')


# In the above table we can see that there are 3 unique Species present in the Iris dataset

# In[6]:


df['Species'].value_counts()


# In[7]:


df.isnull().sum()   #checking for missing values in the dataset


# No null values present in the dataset.

# In[8]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])
df['Species'].value_counts()


# ### Plotting the distribution of species.

# In[9]:


plt.scatter(df['PetalLengthCm'],df['PetalWidthCm'],c=df.Species.values)


# In[10]:


sns.pairplot(df,hue="Species",palette="bright")


# In[11]:


fig=plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),linewidths=1,annot=True)


# In[12]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)


# In[13]:


from sklearn.cluster import KMeans
features = df.iloc[:, [0, 1, 2, 3]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)
    
wcss


# In[14]:


#We will find out the optimum number of clusters using the elbow method
plt.figure(figsize=(10,5))
sns.set(style='whitegrid')
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# We can see that there is a drop between 2 and 3 which implies that performance is higher.

# In[15]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 5)
y_kmeans = kmeans.fit_predict(features)
y_kmeans


# In[16]:


#plotting the centroids and displaying the clusters
fig = plt.figure(figsize=(10, 7))
plt.title('Clusters with Centroids',fontweight ='bold', fontsize=20)
plt.scatter(features[y_kmeans == 0, 0], features[y_kmeans == 0, 1], s = 100, c = 'seagreen', label = 'Iris-versicolour')
plt.scatter(features[y_kmeans == 1, 0], features[y_kmeans == 1, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(features[y_kmeans == 2, 0], features [y_kmeans == 2, 1],s = 100, c = 'yellow', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'red',marker='*', 
            label = 'Centroids')
plt.title('Iris Flower Clusters')
plt.ylabel('Petal Width in cm')
plt.xlabel('Petal Length in cm')
plt.legend()


# In[ ]:




