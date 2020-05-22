# 17.0 简介
# 一个超平面就是n维空间中的n-1维空间。
# 支持向量机需要寻找一个能最大化训练集数据中分类间距的超平面来给数据分类。以一个有两个分类的二维空间为例,我们可以把超平面想象成一条将两个
# 分类隔开的最宽的笔直的"间隔带"(就是一条离两个分类都有一定间距的粗线。)

# 17.1 训练一个线性分类器
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

iris = datasets.load_iris()         # 加载数据,数据里只有两种分类和两个特征
features = iris.data[:100, :2]
target = iris.target[:100]

scaler = StandardScaler()           # 标准化特征
features_standardized = scaler.fit_transform(features)

svc = LinearSVC(C=1.0)              # 创建支持向量分类器

model = svc.fit(features_standardized, target)      # 训练模型
# scikit-learn 的 LinearSVC 实现一个简单的SVC。为了让你对SVC的作用有一个直观的认识,我们在下页中画出了样本点和超平面。
# 尽管SVC在高维空间有很好的表现，但上述解决方案中只加载了两个特征和一部分样本数据，数据集内只有两个分类的数据。因此我们可以可视化这个模型。
# SVC一直试图找到一个能最大化分类之间间距的超平面。

from matplotlib import pyplot as plt
color = ["black" if c == 0 else "lightgrey" for c in target]     # 画出样本点，并且根据其分类上色
plt.scatter(features_standardized[:, 0], features_standardized[:, 1], c=color)

w = svc.coef_[0]    # 获取w       # 创建超平面
a = -w[0]/w[1]      # 斜率
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (svc.intercept_[0]) / w[1]

plt.plot(xx, yy)            # 画出超平面
plt.axis("off"), plt.show()
# 分类0的所有样本点都是黑色的,分类1的所有样本点都是浅灰色的。超平面就是决定新的样本点属于哪一个分类的分界线。
new_observation = [[-2, 3]]     # 创建一个新的样本点
svc.predict(new_observation)    # 预测新样本点的分类
# 对于SVC,有几点需要注意。首先为了方便可视化,我们把本例限定为二元分类问题(只有两个分类),其实SVC在处理多元分类问题时表现也不错。其次,如上图
# 所示,本例的超平面被定义为直线(没有弯曲)。因为我们的数据集是线性可分的,但不幸的是,在真实世界中,这种情况并不多见。
# 更常见的情况是我们不能完美地将数据分类。在这种情况下, 需要在SVC最大化超平面两侧的间距和最小化错误之间取得平衡。
# 在SVC中,最小化错误是通过一个超参数C来控制的。 C是SVC学习器的一个参数，也是学习器将一个样本点分类错误时被施加的罚项。
# 当C很小时SVC可以容忍更多的样本点被错误分类(偏差大，方差小） 当C很大时SVC会因为对数据的错误分类而被重罚,因此反向传播来避免对样本点的错误分类(偏差小，方差大)
# 在scikit-learn中,C是由参数C决定的,默认值是1.0。应该把C当作机器学习算法的一个超参数。可以用12章的模型选择技术进行调校

# 17.2 使用核函数处理线性不可分的数据
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(0)       # 设置随机种子
features = np.random.randn(200, 2)  # 生成两个特征

target_xor = np.logical_xor(features[:, 0] > 0, features[:, 1] > 0)     # 使用异或门(你不需要知道原因)创建线性不可分的数据
target = np.where(target_xor, 0, 1)

svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)       # 创建一个有径向基核函数的支持向量机
model = svc.fit(features, target)

# 简单了解一下支持向量机和核函数的背景
# SVC:    f(x) = β0 + ∑(i∈S)αiK(xi,xi')
# β0是偏差,S是所有支持向量观察值的集合,α是要学习的模型参数,（xi，xi'）是一对支持向量的观察值。最重要的是核函数K，它会比较xi和xi'的相似度，
# 对于K，只需要知道两点。第一:它决定了我们用什么类型的超平面来分离不同的分类。第二:我们使用不同的核函数来创建不同的超平面。
# 如果你想创建基本线性超平面，可以使用线性核函数: K(xi,xi') = ∑(p, j=1)xijxi'j    p是特征的数量。
# 如果我们想获取一个非线性决策边界,可以用多项式核函数: K(xi,xi') = (1+ ∑(d,j=1)xijxi'j)^2    d是多项式核函数的度
# 还可以使用支持向量机中最通用的一种核函数,径向基核函数(radius basis function kernel) K(xi,xi') = e^(-γ∑(P,j=1)(xijxi'j)^2)
# γ是一个超参数，而且必须大于0.
# 要点是：如果数据是线性不可分的，我们可以用其他可选的核函数来替换线性核函数，以创建一个非线性的超平面决策边界。


# 我们可视化一个简单的例子来理解函数背后的逻辑。例子基于 Sebastian Raschka 的工作设计的一个函数，它画出了观察值和一个二维空间里的超平面决策边界。
# 不需要理解这个函数是如何工作的，该函数代码如下
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier):
    cmap = ListedColormap(("red", "blue"))
    xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker="+", label=cl)


svc_linear = SVC(kernel="linear", random_state=0, C=1)  # 创建一个使用线性核函数的SVC
svc_linear.fit(features, target)        # 训练模型
plot_decision_regions(features, target, classifier=svc_linear)     # 画出观察值和超函数
plt.axis("off"), plt.show()
# 如图所示，这个线性超平面的分类效果很差！让我们用一个径向基核函数来替换线性核函数，然后用它来训练一个新模型

svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)   # 创建一个使用径向基核函数的SVC
model = svc.fit(features, target)
plot_decision_regions(features, target, classifier=svc)
plt.axis("off"), plt.show()
# 使用径向基核函数，可以创建一个分类效果比线性核函数好很多的决策边界，这也是在SVC中使用这些核函数的原因。
# scikit-learn中，可以通过设置kernel参数的值来选择想用的核函数。
# 一旦选择了一个核函数，就需要为这个核函数确定一些合适的选项值,比如前面提到的多项式核函数中的d(通过degree参数来设置)和
# 径向基核函数中的γ(通过gamma参数来设置)。我们还需要设置惩罚参数C。在训练模型时, 大部分情况下需要把这些参数视作超参数，然后
# 用模型选择的技术找出能产生性能最佳的模型的参数值组合。

# 17.3 计算预测分类的概率
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
iris = datasets.load_iris()
features = iris.data
target = iris.target

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

svc = SVC(kernel="linear", probability=True, random_state=0)    # 创建SVC对象
model = svc.fit(features_standardized, target)                  # 训练分类器

new_observation = [[.4, .4, .4, .4]]        # 创建一个新的观察值
model.predict_proba(new_observation)        # 查看观察值被预测为不同分类的概率
# SVC算法使用一个超平面来创建决策区间,这种做法并不会直接计算出观察值属于某个分类的概率。但是我们可以输出校准过的分类概率，并给出几点说明。
# 在有两个分类的SVC中可以使用Platt缩放(Platt scaling)。它首先训练这个SVC,然后训练一个独立的交叉验证逻辑回归模型将SVC的输出转换为概率:
# P(y=1|X) = 1/(1+ e^(A*f(x)+B))        A和B是参数向量,f是第i个参数值到超平面的距离，如果数据集不止两个分类，就可以使用Platt缩放扩展
# 计算预测分类的概率有两个主要的问题:第一，因为我们还训练了一个带交叉验证的模型，所以生成预测分类概率的过程会显著增加模型训练的时间；第二
# 因为预测的概率是通过交叉验证计算出来的，所以它们可能不会总是与预测的分类匹配。也就是说，一个观察值可能被预测为属于分类1，但是它被预测为属于分类
# 1的概率却小于0.5。
# 这些预测的概率必须是训练该模型时计算出来的，可以通过设置SVC的probability=True 来做到这一点。在模型训练完之后，可以使用predict_proba 方法
# 输出观察值为每个分类的预测概率。

# 17.4 识别支持向量
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
iris = datasets.load_iris()     # 加载数据,数据中只有两个分类
features = iris.data[:100, :]
target = iris.target[:100]

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

svc = SVC(kernel="linear", random_state=0)      # 创建SVC对象
model = svc.fit(features_standardized, target)     # 训练分类器
model.support_vectors_       # 查看支持向量
# 超平面是由相对而言的一部分观察值(被称为支持向量)所决定的，支持向量机这个名字由此得来。直观地说，你可以把超平面想象是被这些支持向量"举起来"的。
# 因此这些支持向量对于模型来说非常重要。比如，如果从数据集中移除一个非支持向量的观察值,那么模型不会改变;但是如果移除了支持向量，超平面与分类的间距就
# 不会是最大的了。训练完SVC后，scikit-learn 提供了很多识别支持向量的选项。support_vectors_来输出观察值特征的4个支持向量，
model.support_      # 可以用support_来查看支持向量在观察值中的索引值。
model.n_support_    # n_support_查看每个分类有几个支持向量

# 17.5 处理不均衡的分类

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
iris = datasets.load_iris()     # 加载只有两个分类的数据
features = iris.data[:100, :]
target = iris.target[:100]

features = features[40:, :]     # 删除前40个观察值，让各个分类的数据分布不均衡
target = target[40:, :]

target = np.where((target == 0, 0, 1))  # 创建目标向量

scaler = StandardScaler()               # 标准化
features_standardized = scaler.fit_transform(features)

svc = SVC(kernel="linear", class_weight='balanced', C=1.0, random_state=0)  # 创建svc

model = svc.fit(features_standardized, target)
# C 是超参数,决定一个观察值被分错类后的惩罚。支持向量机中处理分类数据不均衡的一个方法是，对不同的分类使用不同的权重C:
# Cj=C*Wj   C是对错误分类的惩罚,wj跟分类j出现的频率反相关,Cj是分类j的C值。 意思是增加对数据少的类别分错类时的惩罚, 来防止模型被数据多的分类占据
# 使用SVC时，可以设置 class_weight='balanced'自动为Cj取值。
