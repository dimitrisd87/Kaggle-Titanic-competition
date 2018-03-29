import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("==========  Pre Processing   ==========")

# import data
train_data = pd.read_csv('titanic_data/train.csv')
test_data = pd.read_csv('titanic_data/test.csv')

# check for missing values
print(train_data.isnull().sum())

# drop columns we don't need  for our model
train_data = train_data.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1)
test_data = test_data.drop(['Cabin', 'Ticket', 'Name'], axis=1)

# replace missing values
datasets = [train_data, test_data]

for dataset in datasets:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

# check that there aren't any missing values for both datasets
mv_list = train_data.isnull().sum()

counter = 0

for x in range(0, mv_list.size):
    # if there is at least one column with missing value, print the whole list and then go out from for loop
    if mv_list[counter] != 0:
        print(train_data.columns[train_data.isnull().any()])
        break
    else:
        counter += 1
        # check if this is the last loop and if it is print a message that there are not missing values
        if x == mv_list.size - 1:
            print('No missing values')

print("==========  END Pre Processing   ==========")
print()
print("==========   Training   ==========")

# set features and target
features = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch', 'Age', 'Fare']
target = ['Survived']

features_for_test = ['PassengerId', 'Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch', 'Age', 'Fare']

# transform variables from categorical to numerical
train_dummy = pd.get_dummies(train_data[features], columns=['SibSp', 'Parch', 'Sex', 'Embarked', 'Pclass'])
test_dummy = pd.get_dummies(test_data[features_for_test], columns=['SibSp', 'Parch', 'Sex', 'Embarked', 'Pclass'])

##
# split the train dataset
new_features = train_dummy.columns.tolist()
train_x, test_x, train_y, test_y = model_selection.train_test_split(train_dummy[new_features], train_data[target],
                                                                    train_size=0.8)

# Select classifier, here RandomForest and fit the model
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train_x[new_features], train_y)

# View a list of the features and their importance scores
list(zip(new_features, clf.feature_importances_))

###
predictions = clf.predict(test_x)
print(predictions)
print("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
print("Test Accuracy  :: ", accuracy_score(test_y, predictions))

# use the test.csv file
test_dummy['Survived'] = clf.predict(test_dummy[new_features])
# Write to csv
test_dummy[['PassengerId','Survived']].to_csv('predictions.csv',index=False)