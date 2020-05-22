# 16.0 简介
# 逻辑回归是一种被广泛是用的有监督分类方法。逻辑回归和它的一些扩展(比如多元逻辑回归)让我们可以通过一个简单易懂的方法来预测一个观察值属于某个分类的概率。

# 16.1 训练二元分类器
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]                  # 加载仅有两个分类的数据

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)      # 标准化特征

logistic_regression = LogisticRegression(random_state=0)    # 创建一个逻辑回归的对象

model = logistic_regression.fit(features_standardized, target)      # 训练模型
# 逻辑回归是一种被广泛使用的二元分类器，在逻辑回归中，线性模型(比如,β0+β1x)被包含在一个逻辑函数 1/(1+e^-z) 也叫sigmoid函数
# P(yi=1|X) = 1/(1+e^(β0+β1x)) 其中P(yi=1|X)是第i个观察值的目标y1属于分类1的概率,X是训练集的数据。β0和β1是要学习的参数,e是欧拉数(Euler's number)
# 逻辑函数的作用是把函数的输出值限定在0和1之间，这样才能被解释为概率。 如果P(yi=1|X)大于0.5, 那么yi的预测分类为分类1，否则为0。

new_observation = [[.5, .5, .5, .5]]        # 创建一个新的观察值
model.predict(new_observation)              # 预测分类
model.predict_proba(new_observation)        # 查看预测的概率

# 16.2 训练多元分类器
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
features = iris.data
target = iris.target

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

logistic_regression = LogisticRegression(random_state=0, multi_class="ovr")     # 创建一对多的逻辑回归对象
model = logistic_regression.fit(features_standardized, target)      # 训练模型
# 独立得看，逻辑回归只是二元分类器。但是，逻辑回归有两个巧妙的拓展可以解决这个问题。
# 第一种拓展是一对多(One vs. Rest,OVR)的逻辑回归，在这种逻辑回归中，
# 对于每一个分类我们都会训练一个单独的模型来判断观察值是不是属于这个分类。(这样又变成二元分类问题了) 它有一个假设，
# 即每一个分类问题(比如，观察值是否为分类0)是相互独立的。
# 第二种拓展是多元逻辑回归(Multinomial Logistic Regression, MLR) 16.1的逻辑函数被一个 softmax 函数替换:
# P(yi=k|X) = e^βk*xi / ∑(K,j=1) e^βj*xi    P(yi=k|X)是第i个观察值yi为类别k的概率，K是分类的数量。
# MLR 有个很实用的优点:它使用predict_proba方法预测概率,更可靠(即这个概率值被更好地校准过)
# 使用LogisticRegression时,可以在这两种技术中选择,默认的选择是OVR(multi_class="ovr")。也可以把multi_class参数设置为multinomial,改为使用MLR

# 16.3 通过正则化来减少方差
from sklearn.linear_model import LogisticRegressionCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
features = iris.data
target = iris.target

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

logistic_regression = LogisticRegressionCV(penalty='l2', Cs=10, random_state=0, n_jobs=-1)
model = logistic_regression.fit(features_standardized, target)      # 训练模型
# 正则化(regularization)是一种通过惩罚复杂模型来减小其方差的方法。就是将一个惩罚项加在我们希望最小化的损失函数上,
# L1惩罚: α∑(p,j=1)|βj|           βj是要学习的p个特征中第j个所对应的参数,α是一个超参数,表示正则化的强度。
# L2惩罚: α∑(p,j=1)βj^2
# α 取值越大,对越大的参数值(也就是更复杂的模型)的惩罚就越重。scikit-learn 遵循常规用法，使用C来代替α，这里C等于正则化强度值的倒数,即 C= 1/α
# 在使用逻辑回归过程中,为了减少方差，我们把C当做需要被调校的超参数。以寻求一个可以创建最佳模型的C值。
# 在scikit-learn中,可以使用LogisticRegressionCV类来有效地调校C。LogisticRegressionCV中参数Cs可以接受两类值
# 一类是用来搜索的C的取值范围(如果提供了一个浮点数列表作为参数的话),另一类是一个整型数值，LogisticRegressionCV会生成一个列表,其长度为这个
# 整形数值，里面的元素是从对数空间中取的大小在-10,000到10,000之间的值。
# 遗憾的是LogisticRegressionCV 不允许我们比较不同的罚项。如果你想这么做必须使用第十二章中提到的模型选择技术。但是效率会低一点。

# 16.4 在超大数据集上训练分类器
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
features = iris.data
target = iris.target

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

logistic_regression = LogisticRegression(random_state=0, solver="sag")      # 创建逻辑回归对象
model = logistic_regression.fit(features_standardized, target)              # 训练模型

# scikit-learn 的 LogisticRegression 提供了不上少用于训练逻辑回归模型的方法，这些方法被称为solver。大部分情况下scikit-learn会帮我们
# 自动选择最佳solver,或者警告我们该solver不能做某些事情。
# 随机平均梯度算法确实可以使我们在超大数据集上比使用其他solver更快地训练出一个模型。
# 需要注意的是，它对特征的单位特别敏感，所以标准化这个步骤很重要。 我们可以在代码中使用 solver="sag"，让学习算法使用这个solver。

# 16.5 处理不均衡的分类
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
features = iris.data
target = iris.target

features = features[40:, :]         # 移除前40个观察值，使分类严重不均匀
target = target[40:]

target = np.where((target == 0), 0, 1)          # 创建目标向量,0代表分类为0，1代表分类除0以外的其他分类。

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

logistic_regression = LogisticRegression(random_state=0, class_weight="balanced")       # 创建逻辑分类器对象
model = logistic_regression.fit(features_standardized, target)                          # 训练模型
# 如果数据特别不均衡可以使用class_weight参数给分类设置权重。确保数据集中的各个分类是均衡的。
# balanced参数值会自动给各分类加上权重，而权重值与分类出现的频率的倒数相关: wj = n/knj
# wj 是分类的权重，n是观察值的数量，nj是属于分类j的观察值数量，k是分类总数。