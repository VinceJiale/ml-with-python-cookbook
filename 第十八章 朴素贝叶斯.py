# 18.0 简介
# 贝叶斯定理是已知事件A发生的先验概率P(A),以及事件A发生的条件下事件B发生的概率P(B|A)时,获得概率P(A|B)的重要方法:
# P(A|B) = P(B|A)*P(A)/P(B)
# 机器学习中,贝叶斯定理在分类上的应用叫做 朴素贝叶斯分类器
# 朴素贝叶斯将机器学习中的一些优点整合到一个分类器中,优点包括:
# 1)方法直观。2)对于小样本量的数据也能起作用。3)进行训练和预测的计算成本低。4)对于各种不同的参数设置总能得到稳定的结果
# 朴素贝叶斯分类器公式:   P(y|x1,...,xj) = P(x1,...,xj|y)*P(y) / P(x1,...,xj)
# P(y|x1,...,xj) 后验概率,表示一个观察值在其j个特征时x1,...,xj的情况下，它的分类是类别y的概率
# P(x1,...,xj|y) 似然概率,表示给定观察值的分类y,其特征是x1,...,xj的概率
# P(y) 先验概率, 使我们在查看数据之前对于分类y出现的概率的猜测
# P(x1,...,xj) 边缘概率

# 朴素贝叶斯中，我们对观察值每一个可能的分类的后验概率进行比较。 后验概率最大的分类就是这个观察值的预测分类
# 对于朴素贝叶斯分类器来说，有两个重要的地方。 1)对于数据的每个特征,必须假定它的似然概率P(xj|y)的统计学分布。最常用的分布有正态(高斯)分布，
# 多项式分布和伯努利分布。分布的选择总是由特征的特性(比如连续,二分等)决定的。 2)朴素贝叶斯得名于一个假设---每个特征和它的似然概率是相互独立的。
# 这种"朴素"经常出错,但是实际操作中它并不影响我们构建一个高平值的分类器

# 18.1 为连续的数据训练分类器
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
iris = datasets.load_iris()
features = iris.data
target = iris.target

classifier = GaussianNB()   # 创建高斯贝叶斯对象
model = classifier.fit(features, target)        # 训练模型

# 高斯朴素贝叶斯分类器中,我们假设对于一个分类为y的观察值,其特征x的似然概率服从正态分布:
# P(xj|y) = 1/sqrt(2π σy^2) * e^(-(xi-μy)^2 / 2σy^2)
new_observation = [[4, 4, 4, 0.4]]   # 创建一个新的观察值
model.predict(new_observation)          # 预测分类

clf = GaussianNB(priors=[0.25, 0.25, 0.5])      # 给定每个分类的先验概率,创建高斯朴素贝叶斯对象   prior参数为一个列表
model = clf.fit(features, target)
# 如果不给priors设定任何值,就会基于我们的数据来计算先验概率, GaussianNB产生的预测概率(predict_proba)是未经过校准的。也就是说是不可信的。
# 如果想生成有用的概率预测，就需要保存回归(isotonic regression) 或者其他相关方法来校正。

# 18.2 为离散数据和计数数据训练分类器
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
text_data = np.array(['I love Brazil. Brazil!',
                      'Brazil is best',
                      'Germany beats both'])        # 创建文本

count = CountVectorizer()                           # 创建词袋
bag_of_words = count.fit_transform(text_data)

features = bag_of_words.toarray()       # 创建特征矩阵
target = np.array([0, 0, 1])            # 创建目标向量

classifier = MultinomialNB(class_prior=[0.25, 0.5])     # 给定每个分类的先验概率，创建一个多项式朴素贝叶斯对象
model = classifier.fit(features, target)    # 训练模型
# 多项式朴素贝叶斯分类器与高斯工作原理相似,但样本数据的特征为服从多项式分布。这种分类器主要用于数据是离散的情况。最常用的场景是文本分类，
# 使用词袋或者TF-IDF的方法。
new_observation = [[0, 0, 0, 1, 0, 1, 0]]       # 创建一个观察值
model.predict(new_observation)
# 如果没有指定class_prior的值会从数据中学习而得到先验概率。但是如果想要用均匀分布计算先验概率的话，可以设置 fit_prior = False
# 含有一个附加的平滑超参数alpha, 它需要被调校。默认值为1.0，设置为0.0意味着不做平滑。

# 18.3 为具有二元特征的数据训练朴素贝叶斯分类器
import numpy as np
from sklearn.naive_bayes import BernoulliNB
features = np.random.randint(2, size=(100, 3))    # 创建3个二元特征
target = np.random.randint(2, size=(100, 1)).ravel()    # 创建一个二元目标向量

classifier = BernoulliNB(class_prior=[0.25, 0.5])   # 给定每个分类的先验概率,创建一个多项式伯努利朴素贝叶斯对象
model = classifier.fit(features,target)
# 伯努利贝叶斯分类器假设所有特征都是二元分类的。与MultinomialNB有类似的超参数alpha,特征。

# 18.4 校准预测概率
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
iris = datasets.load_iris()
features = iris.data
target = iris.target

classifier = GaussianNB()
classifier_sigmoid = CalibratedClassifierCV(classifier, cv=2, method='sigmoid')    # 创建使用Sigmoid校准调校过的交叉验证模型
classifier_sigmoid.fit(features, target)        # 校准概率

new_observation = [[2.6, 2.6, 2.6, 0.4]]        # 创建新的观察值
classifier_sigmoid.predict_proba(new_observation)   # 查看校准过后的概率
# 使用CalibratedClassifierCV类，创建经由k折交叉验证调校过的预测概率。在CalibratedClassifierCV中，训练集被用来训练模型，测试集被用来
# 校准预测概率，返回的预测概率是k折的平均值。

classifier.fit(features, target).predict_proba(new_observation)     # 训练一个高斯朴素贝叶斯分类器来预测观察值的分类概率 得到极端的预测概率

# CalibratedClassifierCV 提供两种校准方法: Platt sigmoid 模型和保存回归。可以通过method参数来设置。保存回归是无参的。样本量很小的情况下，往往会过拟合
