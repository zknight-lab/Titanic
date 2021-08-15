import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import accuracy_score

# 输出最佳参数组合
def report(results, n_top= 3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank:{0}".format(i))
            print("Mean validation score : {0:.3f} (std: {1:.3f})".
                  format(results['mean_test_score'][candidate],
                         results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# 绘制参数优化曲线
def iteration_curve(results):
    results = pd.DataFrame(results)
    results['params_str'] = results.params.apply(str)
    results.drop_duplicates(subset=('params_str', 'iter'), inplace=True)
    mean_scores = results.pivot(index='iter', columns='params_str',
                                values='mean_test_score')
    ax = mean_scores.plot(legend=False, alpha=.6)

    labels = [
        f'iter={i}\nn_samples={shgsearch.n_resources_[i]}\n'
        f'n_candidates={shgsearch.n_candidates_[i]}'
        for i in range(shgsearch.n_iterations_)
    ]

    ax.set_xticks(range(shgsearch.n_iterations_))
    ax.set_xticklabels(labels, rotation=45, multialignment='left')
    ax.set_title('Scores of candidates over iterations')
    ax.set_ylabel('mean test score', fontsize=15)
    ax.set_xlabel('iterations', fontsize=15)
    plt.tight_layout()
    plt.show()

# k折交叉验证训练模型
def build_model(model, X_data, Y_data, n_splits=10, random_state=rng):
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_rf = np.zeros(len(X_data))
    scores = []
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_data, Y_data)):
        print("fold n°{}".format(fold_ + 1))
        clf = model.fit(X_data[trn_idx], Y_data[trn_idx])
        oof_rf[val_idx] = clf.predict(X_data[val_idx])
        scores.append(accuracy_score(Y_data[val_idx], oof_rf[val_idx]))
        print(scores)
    print("score: {:<8.8f}".format(accuracy_score(Y_data, oof_rf)))
    return clf

# 读取数据
rng = 2021
data_for_tree = pd.read_csv('D:/kaggle/titanic/data_for_tree.csv')
train = data_for_tree[date_for_tree['Survived'].notnull()]
test = data_for_tree[data_for_tree['Survived'].isnull()].drop('Survived',axis=1)
X = train.drop('Survived')
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng)

pipe=Pipeline([('select',SelectKBest(k=20)),
               ('classify', RandomForestClassifier(random_state = rng, max_features = 'sqrt'))])

param_test = {'classify__n_estimators':list(range(20,50,2)),
              'classify__max_depth':list(range(3,60,3))}
shgsearch = HalvingGridSearchCV(estimator=pipe, param_grid=param_test, factor=2, min_resources='exhaust', scoring='accuracy' ,random_state=rng, cv=5)

report(shgsearch.cv_results_)
iteration_curve(shgsearch.cv_results_)

model = RandomForestClassifier(random_state = rng, warm_start = True,
                                  n_estimators = 30,
                                  max_depth = 6,
                                  max_features = 'sqrt')

model_rf = build_model(model, X_train, y_train)
y_test_pred = model.predict(X_test)
print('y_test accuracy:', accuracy_score(y_test, y_test_pred))

# 对测试数据进行预测
predictions = model_rf.predict(test)
PassengerId = test.PassengerId
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv('D:/kaggle/titanic/sub.csv', index=False, sep = ',')

