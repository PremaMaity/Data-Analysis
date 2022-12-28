#!/usr/bin/env python
# coding: utf-8

# #EDA with Python and applying Logistic Regression

# In[1]:


##Import Libraries


# In[148]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[149]:


## THE DATA

#lets start reading in the titanic_train.csv file into pandas dataframe


# In[199]:


train=pd.read_csv("C:\\Users\\Prema\\Downloads\\archive.zip")


# In[151]:


train.head()


# In[152]:


train.shape


# In[153]:


## Exploratory Data Analysis
#Let's begin some exploratory data analysis! We'll start by checking out missing data!


# In[154]:


# Missing Data
# We will use seaborn to check our missing data


# In[155]:


train.isnull()


# In[156]:


train.info()


# # Some of the observations wer can check from above that there are 5 columns in integers and 5 category column and 1 in float column.
# # for data analysis we need to convert some columns from int dtype to category dtype.
# 

# In[157]:


train.isnull().sum()


# In[158]:


# from the above obove observations we can check that CABIN column has more than 70 percent data is missing and for AGE column
# more than 20 percent of data is missing and only 2 values are missing in Embarked column.


# In[159]:


sns.heatmap(train.isnull()==True,yticklabels=False)


# # Handle missing Values

# In[160]:


# So from the above observarions we will drop the Cabin column also Passesnger id is unique column we do not really need them.


# In[200]:


train.drop(columns=['Cabin'],inplace=True)
train.drop(columns=['PassengerId'],inplace=True)


# In[201]:


sns.heatmap(train.isnull()==True,yticklabels=False)


# In[202]:


# Now we will impute missing values in Age column


# In[203]:


plt.figure(figsize=(7, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='summer')


# In[204]:


# from the above observartion we can see that the older people are present in class 1 which also make sense also it in passenger 
# 2 the average age decreases and so on. So we will use this method to impute missing values in Age accoring to the Pclass.


# Also we will impute those 2 missing values in Emabrked column with mode.


# In[205]:


def age_impute(age_pclass):
    age=age_pclass[0]
    pclass=age_pclass[1]
    
    if pd.isnull(age):
        
        if pclass==1:
            return 37
    
        elif pclass==2:
            return 30
    
        else:
            return 25
    
    else:
        return(age)
    


# In[206]:


train['Age'] = train[['Age','Pclass']].apply(age_impute,axis=1)


# In[207]:


train['Embarked'].value_counts()
train['Embarked']=train['Embarked'].fillna('S',inplace=True)


# In[208]:


sns.heatmap(train.isnull()==True,yticklabels=False)


# # Yippee! We do not have more missing values.

# In[209]:


# Now we will change some of the interger columns into category columns.

## Survived column
## Pclass


# In[210]:


train['Survived']=train['Survived'].astype('object')
train['Pclass']=train['Pclass'].astype('object')


# In[211]:


train.info()


# # five point Summary

# In[212]:


train.describe()

# From the above description we can conclude that the data is pretty consistent and the average age is 29 and there are few population is below 22 years of age. Maximun age of person present is also 80 years old.
# # Univariate Analysis

# In[213]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)

death_percent=round((train['Survived'].value_counts().values[0]/891)*100)
print('Out of 891 population {} % people died in titanic accident'.format(death_percent))


# In[155]:


sns.set_style('whitegrid')
sns.countplot(x='Pclass',data=train,palette='rainbow')

traveller=print((train['Pclass'].value_counts()/891)*100)
print('\n','55% of population tavelling in passenger class: 3','\n',
      '20 % of population tavelling in passenger class:2','\n',
      '24 % of population tavelling in passenger class:1')


# In[156]:


#From the above we are cocluding that most of the people who have died are actually from Pclass 3 and the people survied
#are from pclass 1.


# In[157]:


plt.figure(figsize=(4,4))
sns.countplot(x='Sex',data=train,palette='RdBu_r')
male_female=print(round((train['Sex'].value_counts()/891*100)))
print('\n','65% of population tavelling is male','\n',
      '35 % of population tavelling is female')


# In[6]:


sns.countplot(train['Embarked'],data=train)
print(round((train['Embarked'].value_counts()/891*100)))
print('\n','72% of population tavelling from S','\n',
      '19 % of population tavelling is C','\n','9% of population are from Q')


# In[5]:


sns.countplot(x='SibSp',data=train,palette='rainbow')
count=print(round(train['SibSp'].value_counts()/891*100))


# # Almost 68% percent of the people are travelling without any spouce and siblings and 23% people have either sibling and spouce.

# In[159]:


plt.figure(figsize=(4,4))
sns.displot(train['Age'].dropna(),bins=40,color='darkblue')

print('people ages between 0 and 20 are :', train[(train['Age']<=20)&(train['Age']>=0)].shape[0])
print('People ages between 20 and 50 are :', train[(train['Age']>20) &(train['Age']<=50)].shape[0])
print('People ages above 50 :', train[(train['Age']>50)].shape[0])


# In[160]:


sns.boxplot(train['Fare'],palette='rainbow')


# In[161]:


train['Fare'].hist(figsize=(5,5),bins=40)
print('people fare between $0 and $50 are :', train[(train['Fare']<=50)&(train['Fare']>=0)].shape[0])
print('people fare between $50 and $100 are :', train[(train['Fare']<=100)&(train['Fare']>50)].shape[0])
print('People ages between $100 and $250 are :', train[(train['Fare']>100) &(train['Fare']<=250)].shape[0])
print('People ages above $250 :', train[(train['Fare']>250)].shape[0])


# In[214]:


train=train[train['Fare']<150]


# In[215]:


train.head()


# In[162]:


## Multivariate Aanalysis


# In[163]:


sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
pd.crosstab(train['Survived'],train['Sex']).apply(lambda x:round((x/x.sum())*100))

# 81% Males died in Titanic accident is way more than female died.


# In[164]:


sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
pd.crosstab(train['Survived'],train['Pclass']).apply(lambda x:round((x/x.sum())*100))

## 76% of people dided in the titanic accident are from passenger class 3.


# In[7]:


sns.countplot(train['Survived'],hue=train['Embarked'],data=train,palette='rainbow')
pd.crosstab(train['Survived'],train['Embarked']).apply(lambda x:round((x/x.sum())*100))

# 66% of the population died are going to city S.


# In[86]:


sns.pairplot(train)
plt.show()


# In[216]:


sns.heatmap(train.corr(),annot=True)


# In[172]:


# From the above we can conclude that there is high corealtion between SibSp and Parch and also Sibsp and Fare as more prople
# travelling more fare will be required aslo Survived and Fare column.


# # Now lets jump into Converting Categorical Features

# In[173]:


# since Embarked column and Sex column are two categorical column we will convert categorical features to 
# dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in 
# those features as inputs.


# In[217]:


Embarked=pd.get_dummies(train['Embarked'],drop_first=True)
Sex= pd.get_dummies(train['Sex'],drop_first=True)


# In[218]:


train.drop(['Embarked','Sex','Name','Ticket'],inplace=True,axis=1)


# In[219]:


train=pd.concat([train,Sex,Embarked],axis=1)


# In[220]:


train.head()


# In[221]:


# Now I feel that there are two columns for parch and Sibsb which both defines family members,lets create one column and include
# both of them with a family size (small,medium,large)


# In[222]:


train['Family']=train['SibSp']+train['Parch']


# In[223]:


train.sample(5)


# In[224]:


def famly_size(number):
    if number==0:
        return 'Alone'
    elif number>=1 and number<=3:
        return 'Medium'
    else:
        return 'Large'


# In[225]:


train['Family']=train['Family'].apply(famly_size)


# In[226]:


train.drop(columns=['SibSp','Parch'],inplace=True)
train.head()


# In[227]:


pd.crosstab(train['Family'],train['Survived']).apply(lambda x:round(x/(x.sum())*100))


# In[228]:


plt.figure(figsize=(6,6))
sns.countplot(x='Survived',hue='Family',data=train)

## Number of people died in accident were alone


# In[229]:


# From above observation we can detech that person who were travelling alone the chaces of survival is very less compares to 
# who are having faimlies with them.


# In[230]:


plt.figure(figsize=(7,7))
sns.heatmap(train.corr(),cmap='summer')


# In[231]:


train['Survived']=train['Survived'].astype('int')
train['Pclass']=train['Pclass'].astype('int')


# In[232]:


Family=pd.get_dummies(train['Family'],drop_first=True)


# In[233]:


train=pd.concat([train,Family],axis=1)
train.drop('Family',axis=1,inplace=True)


# In[234]:


train


# In[238]:


train.info()


# # Yipee your data is ready now!

# In[239]:


# X_train, Y_train,X_test,Y_test split


# In[241]:


X=train.drop(['Survived',],axis=1)
Y=train.Survived


# In[242]:


from sklearn.model_selection import train_test_split


# In[243]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=100)


# In[244]:


x_train.head()


# In[245]:


x_train.info()


# In[246]:


y_train.head()


# In[247]:


x_train.shape,y_train.shape


# In[248]:


# Applying Logistic regression


# In[249]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix


# In[250]:


model=LogisticRegression()


# In[251]:


model.fit(x_train,y_train)


# In[252]:


y_predict=model.predict(x_test)


# In[253]:


y_predict


# In[254]:


accuracy_score(y_test,y_predict)


# In[255]:


confusion_matrix(y_test,y_predict)


# In[ ]:




