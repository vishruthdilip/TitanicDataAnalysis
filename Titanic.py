import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

# Testing Pandas functions
print(train_set.head(5))
#print(train_set.shape)
#print(train_set.index)
#print(train_set.columns)
#print(train_set.info())
#print(train_set.min())
#print(train_set.max())
#print(train_set.mean())
#print(train_set[['Name', 'Age']])
#print(train_set["Age"].mean())
#print(train_set["Age"].median())
#print(train_set["Age"].max())
#print(train_set["Age"].min())
#print(train_set[0:2])

Number_men = train_set[train_set['Sex'].str.match("male")].Sex.count()
Number_women = train_set[train_set['Sex'].str.match("female")].Sex.count()
Percent_men = (Number_men/(Number_women+Number_men))*100
Percent_women = 100-Percent_men
print(Number_men, Percent_men)
print(Number_women, Percent_women)
Men_survived = train_set.loc[(train_set.Sex == 'male') & (train_set.Survived == 1)]
Percent_men_survived = (len(Men_survived)/Number_men)*100
Women_survived = train_set.loc[(train_set.Sex == 'female') & (train_set.Survived == 1)]
Percent_women_survived = (len(Women_survived)/Number_women)*100
print('The percentage of total men survived is %f' %Percent_men_survived)
print('The percentage of total women survived is %f' %Percent_women_survived)
sns.countplot(x='Sex', hue='Survived', data=train_set).set_title('Number of men and women that survived')
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=train_set).set_title('Number of people survived classified by class')
plt.show()
sns.boxplot(x='Sex', y='Age', hue='Survived', data=train_set).set_title('Age variation classified by gender')
plt.show()
sns.histplot(x='Age', bins=20, data=train_set).set_title('Age variation of passengers')
plt.show()
sns.scatterplot(x='Age', y='Fare', y_bins=50, data=train_set) # testing out scatter plot function
plt.show()

#train_set.info()
# Plot graphs to formulate hypotheses
# Data cleaning and preparation
# We can observe that the age is missing for a few people, we can remove these rows from our dataset

train_set.info()
train_set['Age'] = train_set['Age'].fillna(train_set['Age'].mean())
print(train_set['Age'].isnull().sum())
train_set.info()

# we have assigned mean value of age to the missing age values

train_set = train_set.drop(columns='Cabin')
train_set.info()

# we have now also removed the cabins columns as it contains a lot of missing values

train_set['Embarked'] = train_set['Embarked'].fillna(train_set['Embarked'].mode()[0])
print(train_set['Embarked'].isnull().sum())
train_set.info()

# assigned the mode value to null cells under embarked column
train_set_copy= train_set.copy()
train_set_copy.to_csv('correcteddata.csv')

# label encoder to convert categorical data to numerical values for the sake of prediction

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

train_set_copy['Sex'] = le.fit_transform(train_set_copy['Sex'])
# print(train_set_copy['Sex'])
train_set_copy['Embarked'] = le.fit_transform(train_set_copy['Embarked'])
# print(train_set_copy['Embarked'])
#train_set_copy['Sex'] = train_set_copy['Sex'].astype('category')
#train_set_copy['Embarked'] = train_set_copy['Embarked'].astype('category')

train_set_copy = train_set_copy.drop(['Name', 'PassengerId', 'Ticket'], axis = 1)
train_set_x = train_set_copy.drop(['Survived'], axis = 1)
train_set_y = train_set_copy['Survived']
train_set_x.to_csv('trainingsetinput.csv')
print(train_set_x)
print(train_set_y)

# Now let us do the same for the test data, we need the input features for the test
# data so that our model can predict the output i.e whether
# the person survived or not

test_set.info()
test_set['Age'] = test_set['Age'].fillna(test_set['Age'].mean())
print(test_set['Age'].isnull().sum())
test_set['Fare'] = test_set['Fare'].fillna(test_set['Fare'].mean())
test_set['Sex'] = le.fit_transform(test_set['Sex'])
# print(train_set_copy['Sex'])
test_set['Embarked'] = le.fit_transform(test_set['Embarked'])
# print(train_set_copy['Embarked'])
test_set_x = test_set.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis = 1)
test_set_x.info()
print(test_set_x)

# Let us import the true values of the prediction so that we can check the
# accuracy of our logistic regression training model

test_set_y = pd.read_csv('gender_submission.csv')
test_set_y = test_set_y['Survived']

print(test_set_y)
#print(np.isnan(test_set_x))
#print(train_set_x.describe())
#print(test_set_x.describe())

# now the data, both the training data and the test data is cleaned and
# prepared and we can proceed to applying regression models on the data
# applying logistic regression to train the data

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(train_set_x, train_set_y)

# now we can predict the results for our test set
# and check the accuracy

predictions = log_model.predict(test_set_x)
print(log_model.score(test_set_x, test_set_y))

from sklearn.metrics import classification_report
print(classification_report(test_set_y, predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_set_y, predictions))
