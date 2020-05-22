# 15.0 简介
# KNN(K-Nearest Neighbors,K近邻)分类器是有监督学习领域中最简单且被普遍使用的分类器之一。
# KNN分类器一般被认为是一种懒惰(lazy)的学习器,因为严格来说它并没有训练一个模型用来做预测,
# 而是将观察值的分类判断为离它最近的k个观察值中所占比例最大的那个分类。
# 举个例子:如果一个分类未知的观察值周围都是分类为1的观察值，那么它会被认为属于分类1。

# 15.1 找到一个观察值的最近邻
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
features = iris.data

standardizer = StandardScaler()     # 创建standardizer
features_standardized = standardizer.fit_transform(features)    # 特征标准化

nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)     # 两个最近的观察值
new_observation = [1, 1, 1, 1]      # 创建一个观察值

distances, indices = nearest_neighbors.kneighbors([new_observation])   # 获取离观察值最近的两个观察值的索引,以及到这两个点的距离
features_standardized[indices]          # 查看最近的两个观察值
# indices 包括了数据集中最邻近的这两个观察值的位置，X[indices]可以显示它们的数值。
# 从直观上来说，可以把距离看作一种相似度的度量指标，因此这两个最近的观察值代表与我们创建的那朵鸢尾花最相似的两朵鸢尾花。
# 默认情况下,NearestNeighbors 使用minkowski距离，一个超参数p 如果p=1,是manhatten距离。p=2，是euclidean距离。
nearestneighbors_euclidean = NearestNeighbors(
    n_neighbors=2, metric='euclidean').fit(features_standardized)     # 按照欧式距离来算最近的两个邻居
distances       # 查看distances


nearestneighbors_euclidean = NearestNeighbors(
    n_neighbors=3, metric="euclidean").fit(features_standardized)     # 寻找每个观察值和它最近的3个邻居的列表(包括它自己)
nearest_neighbors_with_self = nearestneighbors_euclidean.kneighbors_graph(features_standardized).toarray()  # 每个观察值和它最近的3个邻居的列表(包括它自己)

for i, x in enumerate(nearest_neighbors_with_self):         # 从邻居的列表里移除自己
    x[i] = 0

nearest_neighbors_with_self[0]      # 查看离第一个观察值最近的两个邻居
# 很重要的一件事是转换特征，使所有特征采取同样的单位。这样做是因为距离指标认为所有特征的单位都是相同的。在解决方案中通过StandardScaler来标准化这些特征，解决潜在问题

# 15.2 创建一个KNN 分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)   # 标准化特征

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1).fit(X_std, y)   # 训练一个有5个邻居的KNN分类器
new_observation = [[0.75, 0.75, 0.75, 0.75],
                   [1, 1, 1, 1]]            # 创造两个观察值

knn.predict(new_observation)                # 预测这两个观察值的分类
# 如果给定一个分类未知的观察值，第一步会先基于某个距离指标(比如欧式距离)找到最近的k个观察值(有时候称为xu的邻域),
# 然后这k个观察值基于自己的分类来"投票",得票最多的那个分类就是预测的分类。更正式一点的表述是属于某个分类j的概率是: 1/k * ∑(i∈v)I(yi=j)
# 这里v是xu的邻域内的k个观察值,yi是第i个观察值的分类，I是一个指示函数(即函数数值为1就是真,为0就是假)。
knn.predict_proba(new_observation)      # 查看每个观察值分别属于3个分类中的某一个的概率   概率最高的分类即预测分类。
# KNeighborsClassifier 中一些参数需要注意
# 1)metric: 用来设定使用何种距离指标
# 2)n_jobs: 控制可以使用多少个CPU核。做预测需要计算一个点到数据集中所有点的距离。所以建议使用多核
# 3)algorithm: 用来设定计算最近邻居的算法,尽管不同的算法之间有很大的区别。KNeighborsClassifier 默认自动选择最合适的算法，一般不用为这个参数操心
# 例如我们设定了weights参数,距离近的观察值的投票比距离远的观察值的投票会有更高的权重。    标准化特征在使用KNN分类器之前 非常重要

# 15.3 确定最佳的邻域点集的大小
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
features = iris.data
target = iris.target

standardizer = StandardScaler()
features_standardized = standardizer.fit_transform(features)

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])         # 创建流水线

search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]   # 确定一个可选值的范围

classifier = GridSearchCV(
    pipe, search_space, cv=5, verbose=0).fit(features_standardized, target)   # 创建grid搜索

classifier.best_estimator_.get_params()["knn__n_neighbors"]             # 最佳邻域的大小(k)
# k值的大小对KNN分类器的性能有重要的影响。在机器学习中,我们一直尝试在偏差(bias)和方差(variance)之间找到一种平衡,而k值对这种平衡的影响很明显。
# 如果k=n(这里n是观察值的数量),那么偏差就会很大而方差很小。如果k=1,那么偏差会很小但是方差很大。只有找到了能在偏差和方差之间取得折中的k值, 才能
# 得到最佳的KNN分类器。  在解决方案中,我们用GridSearchCV 对不同k值的KNN分类器做5折交叉验证，可以得到能产生最佳的KNN分类器的k值。

# 15.4 创建一个基于半径的最近邻分类器
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data
target = iris.target

standardizer = StandardScaler()
features_standardized = standardizer.fit_transform(features)

rnn = RadiusNeighborsClassifier(
    radius=.5, n_jobs=1).fit(features_standardized, target)    # 训练一个基于半径的最近邻分类器
new_observations = [[0.75, 0.75, 0.75, 0.75],
                    [1, 1, 1, 1]]   # 创建两个观察值
rnn.predict(new_observations)       # 预测这两个观察值的分类
# 基于半径的最近邻分类器不太常用,其观察值的分类是根据某一半径r范围内所有观察值的分类来预测的。
# 在scikit-learn中, RadiusNeighborsClassifier 与 KNeighborsClassifier 很相似，除了两个参数
# 1)radius: 我们需要指定一个半径来确定某个观察值能不能算作目标观察值的邻居。除非你有很充分的理由要把radius设为某个值,否则最好像对待其他超参数一样
# 在模型选择起见对它进行调整。
# 2)outlier_label: 用来指定如果一个观察值周围没有其他观察值在半径radius的范围内,这个观察值应该被标记为什么。这是一个有用的分辨界外点的方法。

