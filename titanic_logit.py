#!/usr/bin/env python
# coding: utf-8

# In[67]:


#dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# # Collecting Data
# 

# In[2]:


# Read Data
titanic_data = pd.read_csv('train.csv')
test_model = pd.read_csv('test.csv')


# In[3]:


#check data 
titanic_data.head()


# In[4]:


# Length of the data
print("Length or rows in the data is: ",len(titanic_data))


# # Now Analyzing the data

# In[5]:


# Checking the survible rate 0 = not Survived ,1 = Survived
sns.countplot(x = 'Survived',data = titanic_data)


# In[8]:


# Checking male female survived rate , observed that female survived rate is more than male
sns.countplot(x = 'Survived',hue = 'Sex',data = titanic_data)


# In[10]:


# Checking which class survived more observed that class 1 servived more than class 2 and 3
sns.countplot(x='Survived',hue = 'Pclass',data = titanic_data)


# In[13]:


# passengers age ,from the graph it's observed that 20-25 age people travel more
titanic_data['Age'].hist(grid = False)


# In[30]:


# let's see which age range travel in which class,from graph low age people travel in class 3 and high age class 1
sns.boxplot(x=titanic_data['Pclass'],y=titanic_data['Age'])


# In[16]:


# Let's see the fare 
titanic_data['Fare'].hist(grid = False,bins=20)


# # Now clean the data 

# In[18]:


# information about the data
titanic_data.info()


# In[19]:


# Check for null values it's True is data is null False is data present
titanic_data.isnull()


# In[20]:


# Sum all the null values for rows ,from graph we can see the Cabin row have lots of null values so it will
# be better to drop and this Cabin column is not so important
titanic_data.isnull().sum()


# In[21]:


# Drop Cabin column
titanic_data.drop('Cabin',axis = 1,inplace = True)


# In[23]:


# No more Cabin column
titanic_data.head()


# In[26]:


# Another way to visualize the null values,from graph white bars means null values
sns.heatmap(data = titanic_data.isnull(),yticklabels=False)


# In[34]:


# Now fill the Age null value 
titanic_data.dropna(inplace = True)


# In[35]:


len(titanic_data)


# In[37]:


# Again Visualize the data ,from graph you can see no bar for null values
sns.heatmap(data = titanic_data.isnull(),yticklabels=False)


# In[39]:


# You also can check that way
titanic_data.isnull().sum()


# In[40]:


# we can't feed any string values to the model like in this data 'Sex','Name','Embraked'
# here we don't need names so we can drop it


# In[42]:


# Drop name column
titanic_data.drop('Name',axis = 1,inplace=True)


# In[46]:


# Convert String values to numerical
# Sex column 
sex = pd.get_dummies(titanic_data['Sex'])
sex.head()


# In[48]:


# Now see a person could be male or female (Avoid exceptional cases😁)
# we can drop one column from 'Sex' table .It's like
sex = pd.get_dummies(titanic_data['Sex'],drop_first=True)
sex.head()
# if male then 1 ,if female then 0


# In[50]:


# Samething we can do with 'Embarked' and 'Pclass'
# Here there are 3 values different values in this columns ,see data
pclass = pd.get_dummies(titanic_data['Pclass'],drop_first=True)
pclass.head()
# here we can say that if passengers are not travelling into class 2 and 3 then it's obious that it's class 1 (again avoid exceptions😁)


# In[51]:


# from Embarked
embark = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
embark.head()


# In[52]:


# Add this column to the main data
titanic_data = pd.concat([titanic_data,sex,pclass,embark],axis = 1)


# In[53]:


# In last columns you can see this columns are added
titanic_data.head()


# In[59]:


# Now we can drop string columns and unessary columns
titanic_data.drop(['Pclass','Sex','PassengerId','Ticket','Embarked'],axis = 1,inplace = True)


# In[61]:


titanic_data.head()


# # Now Split the data and Train the model

# In[62]:


# x is input data and y is labeled  
x = titanic_data.drop('Survived',axis = 1)
y = titanic_data['Survived']


# In[63]:


# Just for understand
x.head()


# In[64]:


y.head()


# In[70]:


# Now we have to split our data set to train and test for that 
from sklearn.model_selection import train_test_split


# In[71]:


# Split the data in train = 70%,test = 30% ratio
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)


# In[73]:


# you can check this way for every value
x_test.head()


# In[74]:


# Now import Logistic regression model
from sklearn.linear_model import LogisticRegression


# In[75]:


# Create object of the class
logmodel = LogisticRegression()


# In[77]:


# Train the model avoid the warning it,s ok
logmodel.fit(x_train,y_train)


# In[78]:


# Now make Prediction
prediction = logmodel.predict(x_test)


# In[81]:


# If you see prediction then uncomment below code
#print(prediction)


# # Calculate Accuracy

# In[85]:


# To calculate accuracy we have to import confusion_matrix.
# It's nothing but just return a 2D matrix ,from moer check ReadMe file
from sklearn.metrics import confusion_matrix,accuracy_score


# In[86]:


confusion_matrix(y_test,prediction)


# In[97]:


accu = accuracy_score(y_test,prediction)*100
print('Model accuracy is: %.2f'%accu+'%')


# In[96]:


# This way its calculated
ac = (105+63)/(105+21+25+63)*100
print('This way the accuracy is calculated: %.2f'%ac+'%')


# # Our model is ready enjoy

# # Now for testing we will use test_model data

# In[98]:


# Steps will be same for cleaning the data


# In[99]:


test_model.drop('Cabin',axis = 1,inplace = True)


# In[100]:


test_model.dropna(inplace=True)


# In[101]:


test_model.isnull().sum()


# In[102]:


sns.heatmap(test_model.isnull(),yticklabels=False)


# In[103]:


test_model.head()


# In[104]:


sex = pd.get_dummies(test_model['Sex'],drop_first=True)


# In[105]:


pclass = pd.get_dummies(test_model['Pclass'],drop_first=True)


# In[106]:


embark = pd.get_dummies(test_model['Embarked'],drop_first=True)


# In[107]:


test_model = pd.concat([test_model,sex,pclass,embark],axis = 1)


# In[108]:


test_model.drop(['Sex','Pclass','Embarked'],axis = 1,inplace = True)


# In[110]:


test_model.drop(['PassengerId','Name','Ticket'],axis = 1,inplace=True)


# In[111]:


test_model.head()


# In[112]:


# Test the model with this data
prd = logmodel.predict(test_model)


# In[115]:


# Print the predictions
print(prd)
len(prd)


# # 😪 

# In[ ]:




