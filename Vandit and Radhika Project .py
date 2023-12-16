#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# In[2]:


df = pd.read_csv("D:/Business Analytics/Assigments/Semester 3/anemia.csv")


# In[3]:


df.head()


# In[4]:


df.describe().T 


# In[5]:


df.info()


# In[6]:


# Checking the data types in each column
print(df.dtypes)


# In[7]:


# Checking for missing values in each column
print(df.isnull().sum())


# In[8]:


# Creating histograms for each column to visualize distributions
df.hist(bins=15, figsize=(15, 10), layout=(2, 3))
plt.suptitle('Histograms of Each Column')
plt.show()


# In[9]:


# Creating boxplots for each column to detect outliers
df.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False, figsize=(15,10))
plt.show()


# In[27]:


# Function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df.loc[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Remove outliers from the 'Hemoglobin' column
data_cleaned = remove_outliers(df, 'Hemoglobin')


# In[10]:


# Correlation matrix heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()


# In[11]:


# Calculate and print the skewness of each numerical column
skewness = df.skew()
print(skewness)


# In[12]:


# Calculating and printing the counts for each class in the 'Result' column
class_counts = df['Result'].value_counts()
print("Class Counts in 'Result':")
print(class_counts)


# In[13]:


# Calculating and printing the percentage of each class in the 'Result' column
class_percentages = (class_counts / len(df)) * 100
print("\nClass Percentages in 'Result':")
print(class_percentages)


# In[14]:


# if data was imbalanced smote would have been applied here 


# In[15]:


# Feature scaling for continuous variables
scaler = StandardScaler()
features_to_scale = ['Hemoglobin', 'MCH', 'MCHC', 'MCV']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])


# In[16]:


# Features (all columns except 'Result')
X = df.drop('Result', axis=1)  
# Target variable
y = df['Result']               


# In[17]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[18]:


# Model Training
# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train the model on the training data
log_reg.fit(X_train, y_train)

# Model Evaluation
# Predicting on the Test Set
y_pred = log_reg.predict(X_test)

# Generating the confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Printing the evaluation results
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))


# In[26]:


# Perform cross-validation
scores = cross_val_score(log_reg, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Average Cross-Validation Score:", scores.mean())


# In[19]:


from sklearn.svm import SVC

# Initialize the SVM classifier
svm_model = SVC()

# Train the model
svm_model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))


# In[25]:


# Perform cross-validation
scores = cross_val_score(svm_model, X, y, cv=5)

print("Cross-Validation Scores:", scores)
print("Average Cross-Validation Score:", scores.mean())


# In[20]:


# Create the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))


# In[24]:


# Perform cross-validation
# cv specifies the number of folds; 5 or 10 is a common choice
scores = cross_val_score(knn_model, X, y, cv=5)

print("Cross-Validation Scores:", scores)
print("Average Cross-Validation Score:", scores.mean())


# In[22]:


param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




