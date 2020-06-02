# 12.0 简介
# 机器学习中，我们通过最小化某个损失函数的值来训练算法已学习一个模型的参数。此外许多算法还有一些超参数。 超参数必须在学习过程之外定义，超参数
# 的值必须在训练之前设定。 这个过程通常叫做超参数调优(hyperparameter tuning), 超参数优化(hyperparameter optimization) 或者 模型选择
# (model selection)。

# 本书中， 我们将选择最佳学习算法及选择最佳超参数的过程都称为模型选择

# 12.1 使用穷举搜索选择最佳模型
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
features, target = iris.data, iris.target
logistic = linear_model.LogisticRegression()

penalty = ['l1', 'l2']      # 创建正则化惩罚的候选超参数区间
C = np.logspace(0, 4, 10)   # 创建正则化候选超参数区间
hyperparameters = dict(C=C, penalty=penalty)            # 创建候选超参数的字典
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)       # 创建网格搜索对象
best_model = gridsearch.fit(features, target)       # 训练网格搜索

# GridSearchCV 是一种使用交叉验证进行模型选择的暴力方法
# 用户给一个或多个超参数定义候选值的集合。GridSearchCV 使用其中的每个值或值的集合组合来训练模型。性能得分高的模型被选为最佳模型。
# 例如: 本节中的解决方案使用逻辑回归作为学习算法，包含两个超参数 C和正则化惩罚(regularization penalty)
# 我们有10个可能的C值，2个可能的正则化惩罚值，5折交叉验证        10*2*5=100 我们有100中模型候选，从其中选择最佳模型

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])      # 查看最佳超参数
print('Best C:', best_model.best_estimator_.get_params()['C'])
# 默认情况下，在找到最佳超参数之后，GridSearchCV将使用最佳超参数和整个数据集重新训练模型(而不是保留部分数据用于交叉验证)。

best_model.predict(features)        # 预测目标向量

# GridSearchCV中的verbose一般不需要设置。但是在长搜索过程中可以设置该参数来获取一些过程进展信息。verbose参数决定了搜索期间输出的消息量。
# 0代表没有输出，1-3代表输出信息。数字越大表示输出的细节越多。

# 12.2 使用随机搜索选择最佳模型
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV
iris = datasets.load_iris()
features = iris.data
target = iris.target

logistic = linear_model.LogisticRegression()    # 创建逻辑回归对象
penalty = ['l1', 'l2']      # 创建正则化惩罚的候选超参数区间
C = uniform(loc=0, scale=4)     # 创建正则化候选超参数的分布
hyperparameters = dict(C=C, penalty=penalty)    # 创建候选超参数的字典

# 创建随机搜索对象
randomizedsearch = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
# 超参数组合的采样数由 n_iter 迭代次数

best_model = randomizedsearch.fit(features, target)     # 训练随机搜索

# Randomizedsearch 在用户提供的参数分布(如正态分布、均匀分布)上选取特定数量的超参数随机组合。
# 如果指定分布，scikit-learn 将从分布中对超参数进行无放回的随机采样。
# 例如：
uniform(loc=0, scale=4).rvs(10)     # 定义区间(0,4)上的均匀分布，并从中抽取10个样本值
# 如果指定一个列表，将对列表进行有放回的随机抽样
# 例如 本解决方案中的正则化惩罚超参数 ["l1","l2]

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])      # 查看最佳超参数
print('Best C:', best_model.best_estimator_.get_params()['C'])
best_model.predict(features)        # 预测目标向量

# 12.3 从多种学习算法中选择最佳模型
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

np.random.seed(0)   # 设置随机种子
iris = datasets.load_iris()
features = iris.data
target = iris.target

# GridSearchCV()
pipe = Pipeline([("classifier", RandomForestClassifier())])  # 创建流水线

search_space = [{"classifier": [LogisticRegression()],
                 "classifier__penalty":['l1', 'l2'],
                 "classifier__C":np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators":[10, 100, 1000],
                 "classifier__max_features":[1, 2, 3]}]          # 创建候选学习算法及超参数的字典

gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)  # 创建GridSearchCV 对象

best_model = gridsearch.fit(features, target)       # 执行网格搜索

# 如果我们不知道应该用哪种学习算法，可以将学习方法作为搜索的一部分。并定义其中的超函数 使用 "classifier__[hypeparameter_name]这种格式

best_model.best_estimator_.get_params()['classifier']       # 查看最佳模型及其超参数

best_model.predict(features)                                # 预测目标向量

# 12.4 将数据预处理加入模型选择过程
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(0)   # 设置随机数种子

iris = datasets.load_iris()
features = iris.data
target = iris.target

preprocess = FeatureUnion([("std", StandardScaler()),
                           ("pca", PCA())])             # 创建一个包括StandardScaler和PCA的预处理对象
pipe = Pipeline([("preprocess", preprocess),
                 ("classifier", LogisticRegression())])     # 创建一个流水线

search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty":["l1", "l2"],
                 "classifier__C":np.logspace(0, 4, 10)}]        # 创建候选值的取值空间

clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)      # 创建网格搜索对象
best_model = clf.fit(features, target)
best_model.best_estimator_.get_params()['preprocess__pca__n_components']   # 查找最佳模型的主成分数量

# GridSearchCV 使用交叉验证来确定最佳模型。然而交叉验证中，test data 在训练时是不可见的。因此这些数据不应该在预处理<标准化，缩放>中使用，
# FeatureUnion 可以组合多个预处理操作，然后将preprogress一同加入流水线中。最终结果是将拟合,转换和使用各种超参数组合训练模型等这些复杂的操作
# 全部交给scikit-learn 来处理

# 一些预处理有自己的参数，这些参数需要用户提供，比如PCA降维需要用户定义生成转换特征集的主成分数量。幸运的是，scikit-learn 也能简化这个操作，
# 当搜索空间中包含预处理参数选值时，它们会像其他超参数一样被处理。

# 12.5 用并行化加速模型的选择
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
features = iris.data
target = iris.target
logistic = linear_model.LogisticRegression()
penalty = ["l1", "l2"]
C = np.logspace(0, 4, 1000)
hyperparameters = dict(C=C, penalty=penalty)
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)
best_model = gridsearch.fit(features, target)

clf = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=1, verbose=1)        # 创建使用单核的网络搜索
best_model = clf.fit(features, target)

# 12.6 使用针对特定算法的方法加速模型选择
from sklearn import linear_model, datasets
iris = datasets.load_iris()
features = iris.data
target = iris.target
logit = linear_model.LogisticRegressionCV(Cs=100)       # 创建LogisticRegressionCV 对象
logit.fit(features, target)

# 有时候利用学习算法的特性能够比暴力搜索或随机模型搜索更快地找到最佳超参数。在scikit-learn 中， 许多学习算法都有特定的交叉
# 验证方法来利用其自身优势的优势寻找最佳超参数。例LogisticRegressionCV。
# LogisticRegressionCV 方法包含一个Cs参数, 如果Cs为一个列表，就可以从Cs中选择候选超参数。如果Cs是一个整数，就会生成一个含有对应数量
# 候选值的列表。这些候选值按照对数值间隔相等的方式，从0.0001 到 10,000 之间 (C的合理取值范围)抽取。
# 但是 LogisticRegressionCV有一个缺点，它只能搜索C的取值区间。对于scikit-learn中许多模型特有的交叉验证方法来说，这种限制是很常见的。

# 12.7 模型选择后的性能评估
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score
iris = datasets.load_iris()
features = iris.data
target = iris.target
logistic = linear_model.LogisticRegression()
C = np.logspace(0, 4, 20)
hyperparameters = dict(C=C)
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)

cross_val_score(gridsearch, features, target).mean()    # 执行嵌套交叉验证并输出平均得分

# 因为我们使用了交叉验证来评估哪些超参数生成了最佳模型，因此就不能再使用它们评估模型的性能了。
# 解决方法是在 交叉验证中包含另一个用于模型搜索的交叉验证即可。
# 在嵌套交叉验证中，"内部"交叉验证用于选择最佳模型。 而"外部" 交叉验证对模型性能进行无偏估计。

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1)

best_model = gridsearch.fit(features, target)       #[Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    0.2s finished
                                                    # 5*20=100

scores = cross_val_score(gridsearch, features, target)      # 在新的交叉验证中(默认为3折)嵌套了gridsearch