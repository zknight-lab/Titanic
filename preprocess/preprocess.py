import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 读取数据
train_titanic = pd.read_csv('D:/kaggle/titanic/train.csv')
train_titanic.set_index('PassengerId')
test_titanic = pd.read_csv('D:/kaggle/titanic/test.csv')
test_titanic.set_index('PassengerId')

titanic = pd.concat([train_titanic, test_titanic], axis=0, sort=False)

# 处理缺失值
# 根据Pclass，Deck以及Embarked的票价信息，确定将缺失值填补为S
titanic.Embarked = titanic.Embarked.fillna('S')
titanic.Cabin = titanic.Cabin.fillna('Unknown')
titanic[titanic.Sex == 'male'].Age.fillna(titanic[titanic.Sex == 'male'].Age.median())
# 男性与女性在相同的年龄有显著不同的幸存率，故分开填充
female_age = titanic[titanic.Sex == 'female'].Age.median()
male_age = titanic[titanic.Sex == 'male'].Age.median()
titanic.loc[titanic.Age.isnull() & (titanic.Sex == 'female'), 'Age'] = female_age
titanic.loc[titanic.Age.isnull() & (titanic.Sex == 'male'), 'Age'] = male_age

fare = titanic[(titanic.Pclass == 3) & (titanic.Embarked == 'S') & (titanic.Deck == 'U')].Fare.median()
titanic.Fare = titanic.Fare.fillna(fare)

# 构建特征
titanic['Family'] = titanic['SibSp'] + titanic['Parch']
titanic['Deck'] = titanic.Cabin.str.get(0)
# 处理Name属性，Name包含了Title以及Family信息(相同的Family Name对应同一个Family)
titanic['Title'] = titanic.Name.apply(lambda x : x.split(',')[1].split('.')[0].strip())
# 提取家族姓名
titanic['Family_Name'] = titanic.Name.apply(lambda x : x.split(',')[0].strip())
# 根据网上的资料对title进行分组
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady','Jonkheer'], 'Royalty'))  #贵族
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))    #已婚女士
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master'], 'Master'))
titanic.Title = titanic.Title.map(Title_Dict)

titanic.to_csv('D:/kaggle/titanic/data_for_tree.csv', index = 0, sep = ' ')



