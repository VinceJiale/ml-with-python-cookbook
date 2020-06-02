# 11.0 简介
# 训练模型不仅仅是创建一个模型(这很简单)，要创建一个准确的模型(这很难).因此，因此在学习各种模型算法前，需要了解如何评估生成的模型
# 11.1 交叉验证模型
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

digits = datasets.load_digits()     # 加载手写数字的数据集
features = digits.data
target = digits.target
standardizer = StandardScaler()
logit = LogisticRegression()    # 创建逻辑回归对象
pipline = make_pipeline(standardizer, logit)           # 创建包含数据标准化和逻辑回归的流水线

kf = KFold(n_splits=10, shuffle=True, random_state=1)       # 创建k折交叉验证对象

cv_results = cross_val_score(pipline,   # 流水线
                             features,  # 特征矩阵
                             target,    # 目标向量
                             cv=kf,     # 交叉验证方法
                             scoring="accuracy",    # 损失函数
                             n_jobs=-1)     # 使用所有的CPU核
cv_results.mean()   # 计算得分的平均值

# k-折交叉验证(K-Fold Cross-Validation,KFCV) 数据被分成k份,训练时将k-1份数据组合起来作为
# 训练集，剩下的那一份数据作为测试集。将上述过程重复k次，每次取一份不同的数据作为测试集，对模型
# 在k次迭代中的得分取平均值作为总体得分。
cv_results      # 查看10份数据的得分
# array([0.97222222, 0.97777778, 0.95555556, 0.95      , 0.95555556,
#        0.98333333, 0.97777778, 0.96648045, 0.96089385, 0.94972067])

# 在使用KFCV时, 需要考虑3点。1）KFCV假定每个样本都是独立于其他样本的(即数据是独立分布的，Independent Identifically Distributed，IID）
# 如果数据是独立分布的，最好在数据分组将其顺序打乱。可以通过设置shuffle=True 来打乱数据。
# 2）当我们使用KFCV来评估分类器时，通常每一类数据大致平均地分配到k组数据中。例如 目标向量中包括性别，80%样本为男性。则每组数据都包含80%男性，
# 20%女性。这时候可以使用 StratifiedKFold 替换KFold 来使用分层k折交叉分析
# 3）使用Hold-out或交叉验证时，应该基于训练集对数据进行预处理。然后将这些与转换同时应用于训练集和测试集。例如使用standardizer 对数据进行
# 标准化时，我们只需要计算训练集的平均值和方差，然后将这个预处理转换（transform）同时应用于训练集和测试集
from sklearn.model_selection import train_test_split
# 创建训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1, random_state=1)

standardizer.fit(features_train)    # 使用训练集计算标准化参数

features_train_std = standardizer.transform(features_train)     # 将标准化操作应用到训练集和测试集
features_test_std = standardizer.transform(features_test)
# 我们假定测试集是未知的数据，所以如果使用训练集和测试集中的样本一起训练预处理，测试集中的一些信息会在训练过程中泄露。
# 这个规则适用于任何预处理步骤，比如特征选择。

pipline = make_pipeline(standardizer, logit)    # 创建一个流水线
# 有了流水线，使用交叉验证会变得十分简单，首先创造一个预处理数据流水线(如standardizer),然后训练一个模型。

cv_results = cross_val_score(pipline,   # 流水线
                             features,  # 特征矩阵
                             target,    # 目标向量
                             cv=kf,     # 交叉验证方法
                             scoring="accuracy",    # 损失函数
                             n_jobs=-1)     # 使用所有的CPU核
# cv里 k折交叉验证时迄今最常见的方法，也有一些其他方法如 （leave-one-out-cross-validation) 该方法的折数等于样本数
# scoring 参数 指定了衡量模型性能的标准。 本章其他节会讨论
# n_jobs=-1 是使用所有可用的CPU核进行计算。

# 11.2 创建一个基准回归模型
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

boston = load_boston()
features, target = boston.data, boston.target
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)

dummy = DummyRegressor(strategy='mean')     # 创建DummyRegressor对象        简单的基准回归模型
dummy.fit(features_train, target_train)     # 训练回归模型
dummy.score(features_test, target_test)     # 计算R方得分    返回的是R-squared

from sklearn.linear_model import LinearRegression       # 训练自己的模型并与基准模型做比较
ols = LinearRegression()
ols.fit(features_train, target_train)
ols.score(features_test, target_test)

# DummyRegressor 允许我们创建一个简单的模型，以此作为基准和实际的模型进行对比。通常使用这种办法来模拟某个产品或系统中已有的原始预测系统。
# 可选的方法包括训练集的均值或者中位数。此外如果将strategy设置成constant 并使用constant参数。则模型的预测结果都为这个常数。

clf = DummyRegressor(strategy="constant",constant=20)
clf.fit(features_train, target_train)
clf.score(features_test, target_test)   # R-squared 越接近1，代表特征对目标向量的解释越好(即相关性越高)

# 11.3 创建一个基准分类模型
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
iris = load_iris()
features, target = iris.data, iris.target
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)

dummy = DummyClassifier(strategy='uniform', random_state=1)     # 创建 DummyClassifier
dummy.fit(features_train, target_train)
dummy.score(features_test, target_test)     # 基准模型的R方

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(features_train, target_train)
classifier.score(features_test, target_test)

# DummyClassifier 的 strategy 提供了一些生成预测值方法的选项，其中两个特别有用。1)stratified: 使预测结果与训练集中数据类别的比例相同。
# 例如:训练集中20%为女性,则DummyClassifier会有20%概率给出女性作为预测结果。2)uniform:它随机生成均匀的预测。如果样本中20%为女性, uniform
# 算法产生的预测结果为50%女性,50%男性。

# 11.4 评估二元分类器
# Accuracy(准确率)是一种常见的性能指标，表示被正确预测的样本数占参与预测的样本总数的比例: Accuracy = (TP+TN)/(TP+TN+FP+FN)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000,
                           n_features=3,
                           n_informative=3,
                           n_redundant=0,
                           n_classes=2,
                           random_state=1)          # 生成特征矩阵和目标向量
logit = LogisticRegression()                        # 创建逻辑回归对象
cross_val_score(logit, X, y, scoring="accuracy")    # 使用准确率对模型进行交叉验证
# 准确率(Accuracy) 的优点是有直观的解释:被正确预测的样本的百分比
# 缺点为 当存在类别不均衡的情况(现实世界中往往如此), 使用准确率作为衡量的标准会出现模型的准确率高但预测能力不足的情况

# Percision(精准度)是所有被预测为正类的样本中被正确预测的样本的百分比。可以把它看作一种衡量预测结果中噪音的指标, 即当样本被预测为正类时,
# 预测正确的可能性有多大。具有高精准度的模型是悲观的，因为它们仅在非常确定时才会预测样本为正类。   Percision = TP/(TP+FP)
cross_val_score(logit, X, y, scoring="precision")
# recall ratio(召回率)是真阳性样本占所有正类样本的比例。召回率衡量的是模型识别正类样本点能力。召回率高的模型是乐观的。因为他们比较容易
# 将样本预测为正类。 Recall Ratio = TP/ (TP+FN)
cross_val_score(logit, X, y, scoring="recall")
# Percision和Recall Ratio 的缺点为不直观。大多情况下，我们希望在精准度和召回率之间达到某种平衡。而 F1分数可以满足这种需求

# F1 分数是调和平均值(一种用于概率数据的平均数)
# F1=2*(精准度x召回率)/(精准度+召回率)  F1是衡量正类预测的正确程度的指标(Correctness)，代表在被标记为正类的样本中，确实是正类的样本所占的比例
# 分数代表了召回率和精准度之间的一种平衡，两者的相对贡献是相等的。
cross_val_score(logit, X, y, scoring="f1")

# 除了使用cross_val_score，如果已经得到了样本的真实值和预测值。我们还可以直接计算出准确率和召回率等指标
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
y_hat = logit.fit(X_train, y_train).predict(X_test)

accuracy_score(y_test, y_hat)   # 计算准确率

# 11.5 评估二元分类器的阈值
# Receiving Operating Characteristic 受试者工作特征 (ROC) 曲线是评估二元分类器质量的常用方法。ROC Curve 对每一个概率阈值(即用来区分
# 样本属于正类或负类的概率值)比较其真阳性和假阳性的比例。在scikit-learn中，我们可以使用 roc_curve 来计算每个阈值下的真阳性率(TP Rate)和
# 假阳性率(FP Rate),然后用图绘制出来。
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
features, target = make_classification(n_samples=10000,
                                       n_features=10,
                                       n_classes=2,
                                       n_informative=3,
                                       random_state=3)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1, random_state=1)
logit = LogisticRegression()

logit.fit(features_train, target_train)
target_probabilities = logit.predict_proba(features_test)[:, 1]     # 获取预测的概率
false_positive_rate, true_positive_rate, threshold = roc_curve(target_test, target_probabilities)      # 计算真阳性和假阳性的概率

plt.title("Receiver Operating Characteristics")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

logit.predict_proba(features_test)[0:1]     # 获取预测的概率
logit.classes_                              # 查看分类

# 默认情况下，scikit-learn 会将样本预测为概率大于50%(即阈值)的分类。通常，我们会由于某些原因显性地调整阈值而不是使用中间值。例如假阳性对业务
# 的影响很大，我们会更倾向于选用一个较大的概率阈值。这样做虽然会导致有些真阳性的样本被误判，如果样本被预测Wie正类，但是能确信模型的预测基本上是正确的

# TPR（真阳性率）= 真阳性的样本数量/（真阳性的样本数量+假阴性的样本数量）
# FPR（假阳性率) = 假阳性的样本数量/ (假阳性的样本数量+真银性的样本数量)

print("Threshold:", threshold[116])
print("True Positive Rate:", true_positive_rate[116])
print("False Positive Rate", false_positive_rate[116])


print("Threshold:", threshold[45])
print("True Positive Rate:", true_positive_rate[45])
print("False Positive Rate", false_positive_rate[45])       # 提高了 threshold， TOR和FPR 都显著下降

# 因为我们对样本被预测为正类的要求提高了，模型无法识别一些正类文本(TPR较小)，而且错误预测为正类的负类样本也减小了(FPR较小)

# 通过计算ROC曲线下方的面积(AUCROC)来判断在所有可能的阈值下模型的总体性能水平。AUCROC的值越接近1，模型的性能就越好。
roc_auc_score(target_test,target_probabilities)     # 计算ROC 曲线下方的面积

# 11.6 评估多元分类器
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

features, target = make_classification(n_samples=10000,
                                       n_features=3,
                                       n_informative=3,
                                       n_redundant=0,
                                       n_classes=3,
                                       random_state=1)
logit = LogisticRegression()
cross_val_score(logit, features, target, scoring='accuracy')
# 如果分类数据均匀(比如，每个分类的样本数大致相等)，准确率是评估模型性能的一个简单且可解释的指标。准确率是被正确预测的样本数与样本总数之比，
# 并且在二元分类器和多元分类器中的效果一样好。当分类数据不均匀时，应该使用其他评估指标
cross_val_score(logit, features, target, scoring='f1_macro')    # 使用macro-F1 分数对模型进行交叉验证
# _macro 指用来求各分类评估分数的平均值的方式,macro:计算每个分类的得分，然后取加权平均值，每个分类的权值相同。weighted:计算每个分类的得分，
# 然后取加权平均值，权值为每个分类的样本数占总样本数的比例。 micro:计算每个样本分类组合的得分，然后取平均值

# 11.7 分类器性能的可视化
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

iris = datasets.load_iris()
features = iris.data
target = iris.target
class_names = iris.target_names     # 创建目标分类的名称列表
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=1)   # 创建训练集和测试集
classifier = LogisticRegression()
target_predicted = classifier.fit(features_train, target_train).predict(features_test)

matrix = confusion_matrix(target_test, target_predicted)    # 创建混淆矩阵
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)    # 创建一个pandas Dataframe


sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues").set_ylim(3.0, 0)    # 创建热力图
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

# 混淆矩阵是将分类器性能可视化的简单有效的方式。它的主要的优点是很容易解释。矩阵通过以热力图形式呈现，的每一列表示样本的预测分类，而每一行
# 表示样本的真实分类。
# 三点注意事项: 1.一个完美的模型，其混淆军阵应该只有对角线上才有值，而其他位置应该全都为零。糟糕的模型为看起来将样本均匀分配在各单元。
# 2.混淆矩阵不仅可以显示模型将哪些样本的分类预测错了，还可以显示它是怎么分错的，即分类的模式
# 3.对于任何数量的分类，混淆矩阵都适用。如果数据分类太多会不那么直观

# 11.8 评估回归模型
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
features, target = make_regression(n_samples=100,
                                   n_features=3,
                                   n_informative=3,
                                   n_targets=1,
                                   noise=50,
                                   coef=False,
                                   random_state=1)
ols = LinearRegression()
cross_val_score(ols, features, target, scoring="neg_mean_squared_error")    # 使用MSE 对线性回归做交叉验证

cross_val_score(ols, features, target, scoring="r2")                        # 使用决定系数(coefficient of determination)进行交叉验证
# MSE = 1/n* ∑<n,i=1>(yihat-yi)^2 MSE 值越大，总平方误差就越大，模型性能越差。

# 默认情况下，对于scikit-learn的 scoring 参数而言，scoring 值较大的模型优于scoring 值较小的模型。然而 MSE 相反，值越大模型性能越差。
# 因此 scikit-learn 中是用的是 neg_mean_squared_error 参数为 MSE的相反数。

# 决定系数(R2得分)，它代表目标向量变化中有多少能通过模型进行解释:
# R2=1-（ （∑<n,i=1>(yi-yihat)^2) /（∑<n,i=1>(yi-ymean)^2)     R2越接近1，代表模型性能越好

# 11.9 评估聚类模型
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
features, _ = make_blobs(n_samples=1000,
                         n_features=10,
                         centers=2,
                         cluster_std=0.5,
                         shuffle=True,
                         random_state=1)
model = KMeans(n_clusters=2, random_state=1).fit(features)      # 使用KMeans 方法对数据进行聚类，以预测其分类
target_predicted = model.labels_        # 获取预测的分类
silhouette_score(features, target_predicted)        # 评估模型

# 评估有监督学习模型时，要讲预测值(比如，分类值或量化值）与目标向量中对应的真实值进行比较。但是使用聚类方法最常见的原因是数据值没有目标向量。
# 许多评估聚类模型的指标都有要求目标向量，但是当有可用的目标向量时，使用无监督学习(如聚类)方法训练模型可能带来一些不必要的麻烦。

# 如果没有目标向量，就无法评估预测值与真实值之间的差距，不过聚类本身的特性仍能评估。直观地讲，"好"的聚类中同类别样本间的距离非常小(即稠密聚类)，
# 不同类别的样本之间距离非常大(即分离得很彻底)。轮廓系数可以用一个值同时评估这两种特性。第i个样本的轮廓系数的计算公式为:
# si= (bi-ai)/max(ai, bi)
# si 是样本i的轮廓系数，ai是样本i与同类的所有样本间的平均距离，bi是样本i与来自不同分类的最近聚类的所有样本间的平均距离。silhouette_score
# 返回值是所有样本的平均轮廓系数。轮廓系数的值介于-1和1之间。其中1表示内部密集、分离彻底的聚类


# 11.10 创建自定义评估指标
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
features, target = make_regression(n_samples=100,
                                   n_features=3,
                                   random_state=1)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1, random_state=1)


def custom_metric(target_test, target_predicted):
    r2 = r2_score(target_test, target_predicted)    # 计算R方得分
    return r2   # 返回R方得分


score = make_scorer(custom_metric, greater_is_better=True)      # 创建评分函数(评分器)，并定义分数越高代表模型越好

classifier = Ridge()    # 创建岭回归(Ridge Regression) 对象
model = classifier.fit(features_train, target_train)    # 训练岭回归模型

score(model, features_test, target_test)    # 应用自定义评分器
# 使用scikit-learn 里的 make_scorer函数，我们能很轻松地创建自定义指标函数。 定义一个函数接受两个参数（真实目标向量，预测值）并返回一个分数
# make_scorer 创建一个评分器对象，并指定较高的分数代表模型性能较好或者较差(greater_is_better参数)

# 11.11 可视化训练集规模的影响
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
digits = load_digits()
features, target = digits.data, digits.target
# 使用交叉验证为不同规模的训练集计算训练和测试得分
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(),                   # 分类器
                                                        features,
                                                        target,
                                                        cv=10,                                      # 交叉验证折数
                                                        scoring='accuracy',                         # 性能指标
                                                        n_jobs=-1,                                  # 使用所有核
                                                        train_sizes=np.linspace(0.01, 1.0, 50))     # 50个训练集的规模


train_mean = np.mean(train_scores, axis=1)      # 计算训练集得分的平均值和标准差
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)        # 计算测试集得分的平均值和标准差
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,'--',color="#111111", label="Training score")      # 画线
plt.plot(train_sizes, test_mean, color='#111111', label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std,       # 画带状图
                 train_mean + train_std, color='#DDDDDD')
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean +test_std, color='#DDDDDD')

plt.title("Learning Curve")                                 # 创建图
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# 学习曲线将模型在训练集上和交叉验证时的性能(例如准确率，召回率)以及与训练集样本数量之间
# 的关系可视化地表达出来了。这种方法常常被用来判断增加训练集数据规模能否提升模型的性能。

# 11.12 生成对评估指标的报告
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
iris = datasets.load_iris()
features = iris.data
target = iris.target
class_names = iris.target_names
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=1)
classifier = LogisticRegression()

model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)

print(classification_report(target_test,                # 生成分类器的性能报告
                            target_predicted,
                            target_names=class_names))      # 输出中的support指每个分类中的样本数量

# 11.13 可视化超参数值的效果
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

digits = load_digits()
features, target = digits.data, digits.target
param_range = np.arange(1, 250, 2)     # 创建参数的变化范围
train_scores, test_scores = validation_curve(
    RandomForestClassifier(),   # 分类器
    features,                   # 特征矩阵
    target,                     # 目标向量
    param_name='n_estimators',      # 要查看的超参数
    param_range=param_range,        # 超参数值的范围
    cv=3,                   # 交叉验证的折数
    scoring="accuracy",     # 性能指标
    n_jobs=-1       # 使用所有CPU核
)  # 对区间内的参考值分别计算模型在训练集和测试集上的准确率

train_mean = np.mean(train_scores, axis=1)      # 计算训练集得分的平均值和标准差
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)        # 计算测试集得分的平均值和标准差
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, label="Training score", color="black")        # 画出模型在训练集和测试集上的准确率的平均图
plt.plot(param_range, test_mean, label="Cross-validation score", color='dimgrey')

plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")     # 黄处模型在训练集和测试集上的准确率带状图
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

plt.title("Validation Curve with Random Forest")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

# 大多数算法都包含超参数，而且必须在模型训练开始之前就选择好。
# 随机森林的 超参数 为 ：Number of Trees。 大多数情况下，超参数的值是在模型选择阶段确定的。将模型性能与超参数画出来偶尔有用。
# 当数的数量增加到250， 两个准确率分数都没什么变化，表明使用给更多计算资源训练大规模的随机森林可能并不能带来价值
# scikit-learn 中， 我们可以使用 validation_curve 计算验证曲线
# 包括3个重要参数：
# 1）param_name 是需要变化的超参数的名字
# 2）param_range 是超参数取值的区间
# 3）scoring 是模型的评估指标
