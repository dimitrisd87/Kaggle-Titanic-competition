import pandas as pd

print("=====  Pre Processing   =====")

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
    if mv_list[counter] != 0:
        print(train_data.columns[train_data.isnull().any()])
        break
    else:
        counter += 1
        if x == mv_list.size-1:
            print('No missing values')

print("=====  END Pre Processing   =====")