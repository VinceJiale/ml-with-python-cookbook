# 10.0 简介
# 特征选择会保留信息量较高的特征而丢弃信息量较低的特征 (feature selection)
# 特征选择的方法可分成三类: 过滤器、包装器和嵌入式方法
# 过滤器:根据特征点统计信息来选择最优特征
# 包装器:通过不断试错,找出一个可以产生更高质量预测值的特征子集
# 嵌入式:将选择最优特征子集作为机器学习算法训练过程的一部分 (将放到讨论机器学习算法的章节)

# 10.1 数值型特征方差的阈值化
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

iris = datasets.load_iris()
features = iris.data
target = iris.target

thresholder = VarianceThreshold(threshold=.5)   # 创建VarianceThreshold对象
features_high_variance = thresholder.fit_transform(features)    # 创建大方差特征矩阵

features_high_variance[0:3]     # 显示大方差特征矩阵
# 方差阈值化(Variance Thresholding, VT) 是最基本的特征选择方法之一。依据是 小方差的特征可能比大方差的特征的重要性低一些。
# VT 方法的第一步是计算每个特征的方差   算出方差后，方差低于阈值的特征会被丢弃。
# 注意两点：1)方差不是中心化的(单位是特征单位的平方)因此,如果特征数据集中特征的单位不同(例如一个是以年为单位，一个是以美元为单位)那么VT就无法起作用。
# 2) 方差的阈值是手动选择的，所以必须依靠人工来选择一个合适的阈值(或者使用第12章的模型选择方法)。可以通过参数variances_来查看每个特征的方差
thresholder.fit(features).variances_

# 如果特征已经标准化, 则方差阈值将起不到筛选的作用 因为此时平均数为0，方差为1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

selector = VarianceThreshold()
selector.fit(features_std).variances_

# 10.2 二值特征的方差阈值化
from sklearn.feature_selection import VarianceThreshold
# 挑出方差大于给定阈值的二值特征
# 用如下信息来创建特征矩阵
# 特征 0: 80% 为分类 0
# 特征 1: 80% 为分类 1
# 特征 2: 60% 为分类 0， 40% 为分类 1
features=[[0, 1, 0],
          [0, 1, 1],
          [0, 1, 0],
          [0, 1, 1],
          [1, 0, 0]]
thresholder = VarianceThreshold(threshold=(.75*(1 - .75)))      # 创建VarianceThreshold 对象并运行
thresholder.fit_transform(features)
# 在二值特征(即伯努利随机变量)中，方差的计算公式如下:   Var(x) = p(1-p)
# p是观察值属于第一个分类的概率。通过设置p的值,我们可以删除大部分观察值都属于同一个类别的特征

# 10.3 处理高度相对性的特征
import pandas as pd
import numpy as np

features = np.array([[1, 1, 1],
                     [2, 2, 0],
                     [3, 3, 1],
                     [4, 4, 0],
                     [5, 5, 1],
                     [6, 6, 0],
                     [7, 7, 1],
                     [8, 7, 0],
                     [9, 7, 1]])    # 创建一个特征矩阵，其中包括两个高度相关的特征

dataframe = pd.DataFrame(features)  # 将特征矩阵转换成 DataFrame

corr_matrix = dataframe.corr().abs()    # 创建相关矩阵
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))    # 选择相关矩阵的上三角阵
to_drop = [column for column in upper.columns if any(upper[column]> 0.95)]      # 找到相关性大于0.95的特征列的索引

dataframe.drop(dataframe.columns[to_drop], axis=1).head(3)      # 删除特征

# 在机器学习中经常遇到特征高度相关的问题。如果两个特征高度相关，那么他们所包含的信息就会非常相似，因此这两个特征就会存在冗余。解决这个问题
# 的方法很简单:从特征集中删除一个与其他特征高度相关的特征即可

dataframe.corr()        # 相关矩阵
upper       # 相关矩阵的上三角阵
# 最后从每一对高度相关的特征中删除一个



# 10.4 删除与分类任务不相关的特征
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
iris = load_iris()
features = iris.data
target = iris.target

features = features.astype(int)     # 将分类数据转换成整数型数据

# 1) 对分类型数据 计算每个特征和目标向量的卡方统计量(chi-squared)
chi2_selector = SelectKBest(chi2, k=2)      # 选择卡方统计量最大的两个特征
features_kbest = chi2_selector.fit_transform(features, target)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# 2）对数值型数据 计算每个特征和目标向量之间的方差分析F值(ANOVA F-value)
fvalue_selector = SelectKBest(f_classif, k=2)   # 选择F值最大的两个特征
features_kbest = fvalue_selector.fit_transform(features, target)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# 除了选择固定数量的特征(SelectKBest),还可以通过SelectPercentile方法来选择前n%的特征
from sklearn.feature_selection import SelectPercentile
fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# 卡方统计可以检查两个分类向量的相互独立性。卡方统计量代表了观察到的样本数量和期望的样本数量(假设特征与目标向量无关)之间的差异
# 卡方统计的结果是一个数字，它代表观察值和期望值之间的差异。通过计算特征和目标向量的卡方统计值，我们可以得到对两者之间独立性的度量值，
# 如果两者相互独立，就代表特征与目标向量无关，特征中不包括分类所需的信息。相反，如果特征和目标向量高度相关，则代表特征中包含训练分类模型所需的更多信息
# 需要注意的是使用卡方统计量进行选择时，目标向量和特征都必须是分类数据。对于数值型数据，可以将其转换为分类特征后再使用卡方统计。最后使用卡方统计方法时，
# 所有的值必须是非负的。
# 对于数值还可以使用 f_classif 来计算每个特征和目标向量间的ANOVA F值。再根据目标向量对数值型特征分类时，该值可以用来判断每个分类的特征均值之间的差异
# 有多大。例如:如果有一个二元目标向量(性别),和一个数值型特征(考试分数)，那么ANOVA F值可以用来判断男性的平均得分是否与女性的相同。如果相同，那么考试分数
# 并不能帮助我们预测性别，因此这个特征与目标向量是无关的。

# 10.5 递归式特征消除
# 使用scikit-learn 的 RFECV类通过交叉验证（Crossing Validation, CV) 进行递归式特征消除(Recursive Feature Elimination,RFE)
# 该方法会重复训练模型，每一次训练移除一个特征，直到模型性能(例如精度)变差。剩下的特征就是最优特征
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")     # 忽略一些烦人但无害的警告信息

features, target = make_regression(n_samples=10000, n_features=100, n_informative=2, random_state=1)   # 生成特征矩阵，目标向量以及系数

ols = linear_model.LinearRegression()       # 创建线性回归对象
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")      # 递归消除特征 estimator:决定训练模型的类型
rfecv.fit(features, target)               # step:决定每次迭代都丢弃的特征的数量或比例 scoring:决定在做交叉验证时评估模型性能的方法
rfecv.transform(features)

rfecv.n_features_   # 最优特征的数量
rfecv.support_      # 哪些特征时最优特征
rfecv.ranking_      # 将特征从最好(1)到最差排序


