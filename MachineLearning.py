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
feature_file = pd.read_excel("data.xlsx")

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



Cross_Validation = True

##############################模型
# xgboost
from xgboost import XGBClassifier
xgbc_model=XGBClassifier()

# 随机森林
from sklearn.ensemble import RandomForestClassifier
rfc_model=RandomForestClassifier()

# ET
from sklearn.ensemble import ExtraTreesClassifier
et_model=ExtraTreesClassifier()

# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
gnb_model=GaussianNB()

#K最近邻
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier()

#逻辑回归
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()

#决策树
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()

#支持向量机
from sklearn.svm import SVC
svc_model=SVC()


if Cross_Validation:
    # xgboost
    xgbc_model.fit(x,y)

    # 随机森林
    rfc_model.fit(x,y)

    # ET
    et_model.fit(x,y)

    # 朴素贝叶斯
    gnb_model.fit(x,y)
    
    # K最近邻
    knn_model.fit(x,y)
    
    # 逻辑回归
    lr_model.fit(x,y)
    
    # 决策树
    dt_model.fit(x,y)
    
    # 支持向量机
    svc_model.fit(x,y)

    from sklearn.cross_validation import cross_val_score
    print("\n使用５折交叉验证方法得随机森林模型的准确率（每次迭代的准确率的均值）：")
    print("\tXGBoost模型：",cross_val_score(xgbc_model,x,y,cv=5).mean())
    print("\t随机森林模型：",cross_val_score(rfc_model,x,y,cv=5).mean())
    print("\tET模型：",cross_val_score(et_model,x,y,cv=5).mean())
    print("\t高斯朴素贝叶斯模型：",cross_val_score(gnb_model,x,y,cv=5).mean())
    print("\tK最近邻模型：",cross_val_score(knn_model,x,y,cv=5).mean())
    print("\t逻辑回归：",cross_val_score(lr_model,x,y,cv=5).mean())
    print("\t决策树：",cross_val_score(dt_model,x,y,cv=5).mean())
    print("\t支持向量机：",cross_val_score(svc_model,x,y,cv=5).mean())

# 使用交叉验证在Xgboost、随机森林、ET、朴素贝叶斯模型的准确率

