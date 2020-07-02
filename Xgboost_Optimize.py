import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
warnings.filterwarnings(action ='ignore',category = DeprecationWarning)
# 加载样本数据集
# feature_file = pd.read_excel("data.xlsx")

x = []# 特征数据
y = []# 标签
for index in feature_file.index.values:
    # print('index', index)
    # print(feature_file.ix[index].values) 
    x.append(feature_file.ix[index].values[1: -1]) # 每一行都是ID+特征+Label
    y.append(feature_file.ix[index].values[-1] - 1) #
x, y = np.array(x), np.array(y)
print('x,y shape', np.array(x).shape, np.array(y).shape)
print('样本数', len(feature_file.index.values))
# 分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=12343)
print('训练集和测试集 shape', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#
max_depth_min_child_weight = True
gamma = True
subsample_colsample_bytree = True
reg_alpha = True
reg_lambda = True
Grid_Search = True
step6 = False

'''
Xgboost参数调优的一般方法
    调参步骤：
　　1、学习速率（learning rate）。在0.05~0.3之间波动，通常首先设置为0.1。
　　2、进行决策树特定参数调优（max_depth , min_child_weight , gamma , subsample,colsample_bytree）在确定一棵树的过程中，我们可以选择不同的参数。
　　3、正则化参数的调优。（lambda , alpha）。这些参数可以降低模型的复杂度，从而提高模型的表现。
　　4、降低学习速率，确定理想参数。
'''
if max_depth_min_child_weight:
    #max_depth和min_child_weight参数调优
    # max_depth和min_child_weight参数对最终结果有很大的影响。max_depth通常在3-10之间，min_child_weight。采用栅格搜索（grid search），我们先大范围地粗略参数，然后再小范围的微调。
    # 网格搜索scoring = 'roc_auc' 只支持二分类，多分类需要修改scoring（默认支持多分类）
    
    param_test1 = {
     'max_depth':range(3,10,2),
     'min_child_weight':range(1,6,2)
    }
    param_test2 = {
    'max_depth':[4,5,6],
    'min_child_weight':[4,5,6]
    }
    from sklearn import svm, grid_search, datasets
    from sklearn import grid_search
    gsearch = grid_search.GridSearchCV(
    estimator = XGBClassifier(
    learning_rate =0.1,
    n_estimators=140, max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid = param_test2,
    scoring='roc_auc',
    n_jobs=4,
    iid=False,
    cv=5)
    gsearch.fit(X_train,y_train)
    print('max_depth_min_child_weight')
    print('gsearch1.grid_scores_', gsearch.grid_scores_)
    print('gsearch1.best_params_', gsearch.best_params_)
    print('gsearch1.best_score_', gsearch.best_score_)

if gamma:
    # gamma参数调优
    # 　　在已经调整好其他参数的基础上，我们可以进行gamma参数的调优了。Gamma参数取值范围很大，这里我们设置为5，其实你也可以取更精确的gamma值。
    
    from sklearn import svm, grid_search, datasets
    from sklearn import grid_search
    param_test3 = {
     'gamma':[i/10.0 for i in range(0,5)]
    }
    gsearch = grid_search.GridSearchCV(
    estimator = XGBClassifier(
    learning_rate =0.1,
    n_estimators=140,
    max_depth=4,
    min_child_weight=5,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid = param_test3,
    scoring='roc_auc',
    n_jobs=4,
    iid=False,
    cv=5)
    gsearch.fit(X_train,y_train)
    print('gamma')
    print('gsearch1.grid_scores_', gsearch.grid_scores_)
    print('gsearch1.best_params_', gsearch.best_params_)
    print('gsearch1.best_score_', gsearch.best_score_)

if subsample_colsample_bytree:
    #调整subsample 和 colsample_bytree参数
    # 　　尝试不同的subsample 和 colsample_bytree 参数。我们分两个阶段来进行这个步骤。这两个步骤都取0.6,0.7,0.8,0.9作为起始值。
    #取0.6,0.7,0.8,0.9作为起始值
    from sklearn import svm, grid_search, datasets
    from sklearn import grid_search
    param_test4 = {
     'subsample':[i/10.0 for i in range(6,10)],
     'colsample_bytree':[i/10.0 for i in range(6,10)]
    }
      
    gsearch = grid_search.GridSearchCV(
    estimator = XGBClassifier(
    learning_rate =0.1,
    n_estimators=177,
    max_depth=4,
    min_child_weight=5,
    gamma=0.4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid = param_test4,
    scoring='roc_auc',
    n_jobs=4,
    iid=False,
    cv=5)
    gsearch.fit(X_train,y_train)
    print('subsample_colsample_bytree------------------')
    print('gsearch1.grid_scores_', gsearch.grid_scores_)
    print('gsearch1.best_params_', gsearch.best_params_)
    print('gsearch1.best_score_', gsearch.best_score_)

if reg_alpha:
    #正则化参数调优reg_alpha
    # 　　由于gamma函数提供了一种更加有效的降低过拟合的方法，大部分人很少会用到这个参数，但是我们可以尝试用一下这个参数。
    from sklearn import svm, grid_search, datasets
    from sklearn import grid_search
    param_test6 = {
     'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
    }

    gsearch = grid_search.GridSearchCV(
    estimator = XGBClassifier(
    learning_rate =0.1,
    n_estimators=177,
    max_depth=4,
    min_child_weight=5,
    gamma=0.4,
    subsample=0.9,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid = param_test6,
    scoring='roc_auc',
    n_jobs=4,
    iid=False,
    cv=5)
    gsearch.fit(X_train,y_train)
    print('reg_alpha------------------')
    print('gsearch1.grid_scores_', gsearch.grid_scores_)
    print('gsearch1.best_params_', gsearch.best_params_)
    print('gsearch1.best_score_', gsearch.best_score_)

if reg_lambda:
    #正则化参数调优reg_lambda
    # 　　由于gamma函数提供了一种更加有效的降低过拟合的方法，大部分人很少会用到这个参数，但是我们可以尝试用一下这个参数。
    from sklearn import svm, grid_search, datasets
    from sklearn import grid_search
    param_test7 = {
     'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
    }
    gsearch = grid_search.GridSearchCV(
    estimator = XGBClassifier(
    learning_rate =0.1,
    n_estimators=177,
    max_depth=4,
    min_child_weight=5,
    gamma=0.4,
    subsample=0.9,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid = param_test7,
    scoring='roc_auc',
    n_jobs=4,
    iid=False,
    cv=5)
    gsearch.fit(X_train,y_train)
    print('reg_lambda------------------')
    print('gsearch1.grid_scores_', gsearch.grid_scores_)
    print('gsearch1.best_params_', gsearch.best_params_)
    print('gsearch1.best_score_', gsearch.best_score_)
    
