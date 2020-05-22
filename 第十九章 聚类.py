# 19.0 简介
# 聚类为无监督学习。聚类算法的目标是找出这些观察值潜在的分类。如果做得好的话，我们能在无目标向量的情况下预测观察值的分类。
# 聚类算法很多，它们使用了多种不同的方法来识别数据中的聚类。

# 19.1 使用K-Means 聚类算法
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
iris = datasets.load_iris()
features = iris.data

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

cluster = KMeans(n_clusters=3, random_state=0, n_jobs=-1)   # 创建K-Means对象
model = cluster.fit(features_std)   # 训练模型
# K-Means 聚类是最常见的一张聚类算法。在K-Means 聚类中，算法试图把观察值分到k个组中,每个组的方查都查不多。分组的数量k是用户设置的一个超参数。具体来讲
# K-Means 算法有如下几个步骤
# 1. 随机创建k个分组(cluster)的"中心"点
# 2. 对于每个观察值:1)算出每个观察值和这k个中心点之间的距离。2)将观察值指派到离它最近的中心点的分组。
# 3. 将中心点移动到相应分组的点的平均值位置
# 4. 重复步骤2，3 直到没有观察值需要改变它的分组。这时该算法被认为已经收敛，而且可以停止了。

# 关于K-Means算法有3点值得注意
# (1)K-Means聚类假设所有的聚类都是凸形的(比如圆形或球形) (2)所有特征在同一度量范围内,可以使用标准化特征。
# (3)分组之间是均衡的(即每个分组中观察值的数量大致相等)         如果觉得无法满足这些假设,就可能需要尝试其他的聚类方法。
# 我们往往不知道k(n-cluster)的数量，这种情况下我们可以基于一些条件来选择k。例如轮廓系数(sihouette coefficient 详见11.9)

model.labels_       # 查看预测分类

iris.target         # 真实分类
# 如果分类的数量设置错了,K-Means 的表现就会急剧下降
new_observation = [[0.8, 0.8, 0.8, 0.8]]        # 创建新的观察值
model.predict(new_observation)      # 预测观察值的分类
# 这个观察值被预测为离某个分类的中心点最近的分类，可以使用cluster_centers_ 来查看这些中心点
model.cluster_centers_

# 19.2 加速 K-Means 聚类
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
iris = datasets.load_iris()
features = iris.data

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

cluster = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=100)     # 创建 K-Means 对象
model = cluster.fit(features_std)       # 训练模型
# Mini-Batch K-Means 和 K-Means 算法的工作原理类似。如果不深究过多细节，那么算法区别是 Mini-Batch 计算量最大的步骤只是在观察值的一部分
# 随机样本上而非所有的观察值上执行。这个方法在只损失一部分质量的情况下显著缩短算法收敛的时间。 MiniBatchKMeans的用法和KMeans十分相似，
# 最大的区别在于batch_size参数。batch_size 控制每个批次中随机选择的观察值的数量。批次中观察值越多，在训练过程中需要花费的算力就越大。

# 19.3 使用 Meanshift 聚类算法
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
iris = datasets.load_iris()
features = iris.data

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

cluster = MeanShift(n_jobs=1)      # 创建Meanshift对象
model = cluster.fit(features_std)   # 训练模型
# K-Means算法有一个缺点,这就是要在训练之前设定聚类的数量k,而且还要假设聚类的形状。Meanshift算法没有这些限制。
# Meanshift是一个简单的概念，用类比的方法来解释，想象有一个有雾气弥漫的足球场(即二维空间),上面站着100个人(即我们的观察值)。因为雾很大,
# 人只能看到很近的地方。每分钟每个人向四周看一看,然后朝着可以看到最多人的方向移动一步。随着时间流逝,因为人们一次次地朝着越来越大的人群移动，
# 球场上的人开始聚集成一个个小组，最终这些人就在球场上形成了一个聚类。每个人的分类被指定为他们最终所在的聚类。

# MeanShift 有两个重要的参数。第一个是bandwidth，设定了一个观察值用以决定移动方向的区域(又叫作核)的半径。在我们的类比重，bandwidth就是
# 一个人能在雾里能看到的距离。我们可以手动设定这个参数，默认情况下,Meanshift会自动估计一个合理的bandwidth值(会显著增加计算开销) 第二,
# 有时候执行Meanshift算法时，在一个观察值的核里看不到任何其他观察值。相当于在球场上有一个人看不到任何其他人。默认情况写下，meanshift把所有
# 的这些"孤儿"观察值分配给离它最近的观察值的核。如果你想丢弃这些孤值,可以设置cluster_all=False,这样所有的孤值的标签就被设定为-1.

# 19.4 使用DBSCAN聚类算法
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
iris = datasets.load_iris()
features = iris.data

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

cluster = DBSCAN(n_jobs=1)     # 创建DBSCAN对象
model = cluster.fit(features_std)       # 训练模型
# 聚类是很多观察值紧密聚集在一起的区域。DBSCAN算法就是受这一点启发而来的。它对于聚类的形状没有任何假设。DBSCAN算法有如下几步:
# 1.先选择一个随机的观察值。
# 2.如果xi的近邻值为最小限度数量的话,就把它归入一个聚类
# 3.对xi的所有邻居重复执行步骤2,对邻居的邻居也如此。以此类推。这些点就是聚类的核心观察值。
# 4.一旦步骤3处理完所有邻近的观察值,就选择一个新的随机点(重新开始执行步骤1)
# 一旦完成这些步骤,我们就会得到一个聚类的核心观察值的合集。
# 最后,凡是在聚类附近但又不是核心的观察值将被认为属于这个聚类,而那些离聚类很远的观察值将会被标记为噪音。
# 3个参数:
# eps: 从一个观察值到另一个观察值的最远距离,超过这个距离将不再认为二者是邻居
# min_samples: 最小限度的邻居数量,如果一个观察值在其周围小于eps距离的范围内有超过这个数量的邻居,就被认为是核心观察值。
# metric: eps所用的距离度量，比如minkowski或者euclidean，如果使用minkowski 可以用参数p来设定幂次。
# 如果观察训练集数据中的聚类，可以看到有两种聚类识别出来，被标记为0，1 噪音观察值被标记为-1。

model.labels_       # 显示聚类情况

# 19.5 使用层次合并聚类算法
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

iris = datasets.load_iris()
features = iris.data

scaler = StandardScaler()
features_std = scaler.fit_transform(features)
cluster = AgglomerativeClustering(n_clusters=3)     # 创建一个Agglomerative聚类对象
model = cluster.fit(features_std)

# Agglomerative 聚类是一个强大的、灵活的层次聚类算法。在Agglomerative 聚类中,所有观察值一开始都是一个独立的聚类。接着满足一定条件的聚类
# 被合并。不断重复合并过程,让聚类不断增长,直到达到某个临界点。AgglomerativeClustering使用linkage参数来决定合并策略,使其可以最小化下面的值
# 1.合并后的聚类的方差(ward)
# 2.两个聚类之间观察值的平均距离(average)
# 3.两个聚类之间的观察值的最大距离(complete)

# 另外两个有用的参数
# affinity 决定linkage使用何种距离度量(比如minkowski或者euclidean等).
# n_clusters 设定了聚类算法试图寻找的聚类数量   即 直到有了n_clusters个聚类时,聚类的合并才结束。
model.labels_       # 显示聚类的情况
