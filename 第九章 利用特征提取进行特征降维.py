# 9.0 简介
# 并非所有特征都是同样重要的，通过特征提取（feature extraction) 进行降维的目的就是对原特征集P<original>进行变换，变换后
# 得到的新特征集记为P<new>,P<original> > P<new>, 但P<new> 中保留了源特征集的大部分信息。换句话说就是通过牺牲一小部分数据
# 信息来减少特征点数量，并保证还能做出准确的预测。  存在一个缺点: 即产生的新特征对人类而言是不可解释的。新特征拥有训练模型的能力，
# 但是在人类看来它们却像是一些随机数字段集合。如果你希望能解释模型，可以采用特征选择来进行特征降维

# 9.1 使用主成分进行特征降维
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()     # 加载数据
features = StandardScaler().fit_transform(digits.data)      # 标准化特征矩阵

pca = PCA(n_components=0.99, whiten=True)   # 创建可以保留99%信息量(用方差表示的）PCA
# PCA参数: n_components 有两种含义:1)由具体的参数值决定。如果值大于1,则n_components 将会返回和这个值相同数量的特征，将会带来一个问题
# 如何选择最合适的特征数量。 2)值在0和1之间，pca就会维持一定信息量(在算法中,用方差代表信息量)的最少特征数。 通常情况下,n_components 取值
# 0.95 或 0.99,意味着保留95%或99%的原始特征信息量。参数whiten=True,表示对每一个主成分都进行转换以保证他们的平均值为0, 方差为1。另一个
# 参数是svd_solver="randomized",代表使用随机方法找到第一个主成分。(这种方法通常速度很快)


features_pca = pca.fit_transform(features)   # 执行PCA

print("Original number of features:", features.shape[1])        # 显示结果， 从输出结果来看PCA保留了矩阵99%的信息同时将特征数量减少了10个
print("Reduced number of features:", features_pca.shape[1])
# 主成分分析法（Principal Component Analysis, PCA) 是一种流行的线性降维方法。PCA将样本数据映射到特征矩阵的主成分空间 (主成分
# 空间保留了大部分的数据差异，一般具有更低的维度）。PCA 是一种无监督学习方法，也就是说它只考虑特征矩阵而不需要目标向量的信息。

# 9.2 对线性不可分数据进行特征降维
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

features, _ = make_circles(n_samples=1000, random_state=1, noise=0.1, factor= 0.1)   # 创建线性不可分数据
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)   # 应用基于径向基函数(Radius Basis Function,RBF)核的Kernel PCA方法
features_kpca = kpca.fit_transform(features)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kpca.shape[1])

# 理想情况下我们希望维度变换既能降低数据的维度,又可以使数据变得线性可分，Kernel PCA可以做到这两点
# 核(Kernel)能够将线性不可分数据映射到更高的维度，数据在这个维度是线性可分的，我们把这种方式叫做核机制(Kernel trick)
# KernelPCA 提供了很多可用的核函数,可以使用参数kernel来指定。常用的核函数是 高斯径向基函数(rdf),其他核函数还有多项式核(Poly)
# 和 sigmoid核(sigmoid)。我们甚至可以制定一个线性映射，利用它可以得到与标准PCA相同的结果。

# KernelPCA 必须制定参数的数量(例如,n_components = 1), 此外，每个核都有自己的超参数需要设置，例如径向基函数需要设置gamma值。
# 如何设置这些值需要使用不同的核函数和参数值反复训练机器学习模型，找出产生最优模型的参数组合。

# 9.3 通过最大化类间可分性进行特征降维
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()     # 加载Iris flower 数据集
features = iris.data
target = iris.target

lda = LinearDiscriminantAnalysis(n_components=1)            # 创建并运行LDA，然后用它对特征做变换
features_lda = lda.fit(features, target).transform(features)

print("Original number of features:",features.shape[1])
print("Reduced number of features:",features_lda.shape[1])

lda.explained_variance_ratio_   # 可以使用此参数来查看每个成分保留的信息量 在本解决方案中，单个成分保留了99%的信息量

# LDA是一种分类方法，也是常用的降维方法。LDA与PCA原理类似。它将特征空间映射到较低维度的空间，然而在PCA中，只需关注使数据差异最大化的成分轴,
# 在LDA中，我们另一个目标是：找到使类间差异最大的成分轴。
# n_components 表示的是 需要返回的特征数量。为了找到n_components的最优值，可以参考 explained_variance_ratio 的输出。

# 具体来说可以用 LinearDiscriminantAnalysis(n_components=None) 返回每个成分特征保留的信息量的百分比，然后计算需要多少个成分特征才能保留高于
# 阈值（通常为0.95 或 0.99) 的信息量：

lda = LinearDiscriminantAnalysis(n_components=None)     # 创建并运行LDA
features_lda = lda.fit(features, target)

lda_var_ratios = lda.explained_variance_ratio_          # 获取方差百分比的数组

def select_n_components(var_ratio, goal_var:float) -> int:  # 函数定义
    total_variance = 0.0    # 设置总方差的初始值
    n_components = 0        # 设置特征数量的初始值
    for explained_variance in var_ratio:    # 遍历方差百分比数组的元素
        total_variance += explained_variance    # 将该百分比加入总方差
        n_components += 1   # n_components 的值加1
        if total_variance >= goal_var:  # 如果达到目标阈值
            break
    return n_components


select_n_components(lda_var_ratios, 0.95)    # 运行函数

# 9.4 使用矩阵分解法进行特征降维
from sklearn.decomposition import NMF
from sklearn import datasets

digits = datasets.load_digits()
features = digits.data

nmf = NMF(n_components=10, random_state=1)  # 创建NMF，进行转换并应用
features_nmf = nmf.fit_transform(features)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_nmf.shape[1])

# NMF(Non-negative Matrix Factorization)非负矩阵分解法 是一种无监督的线性降维方法, 它可以分解矩阵(将特征矩阵分解为多个矩阵，其乘积近似于原始矩阵), 将特征矩阵转换
# 为表示样本与特征之间潜在关系的矩阵。简单地说，NMF 可以减少维度，因为在矩阵乘法中，两个因子(相乘的矩阵)的维度要比得到
# 的乘积矩阵的维数低得多。正式地，给定一个期望的返回特征数量，NMF将把特征矩阵分解为: V≈WH  V（d x n) d个特征，n个样本。
# W (d x r），H（r x n）.前提是: 特征矩阵中不能包含负数值。此外NMF 不会告诉我们保留了原始数据的信息量。因此找出n_components 的
# 最优值的最佳方法就是不断尝试一系列可能的值。直到找出能生成最佳学习的模型的值

# 9.5 对稀疏数据进行特征降维
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets


digits = datasets.load_digits()
features = StandardScaler().fit_transform(digits.data)  # 标准化特征矩阵
features_sparse = csr_matrix(features)  # 生成稀疏矩阵

tsvd = TruncatedSVD(n_components=10)    # 创建 tsvd
features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)    # 在稀疏矩阵上执行TSVD

print("Original number of features:", features_sparse.shape[1])
print("Reduced number of features:", features_sparse_tsvd.shape[1])
# TSVD (Truncated Singular Value Decomposition) 截断奇异值分解。在常规SVD中, 对于给定的d个特征，SVD将创建
# d x d 维的因子矩阵, 而 TSVD 将返回 n x n 维的因子矩阵，其中n是预先指定的参数。与PCA相比，TSVD的优势在于它适用于稀疏矩阵
# TSVD的问题在于 其输出值的符号会在多次拟合中不断变化 (这是由其使用随机数生成器的方式决定的）。一个简单的解决方法是对每个
# 预处理管道只是用一次fit方法，然后多次使用transform方法。 TSVD 需要通过n_components 指定想要输出的特征数。寻找最佳特征数
# 是一种方法在模型选择时将n_components 作为超参数进行优化。由于 TSVD 提供了每个成分保留的原始特征矩阵信息比例，因而我们也可以
# 按照要保留的信息量(常用的值是95%,99%)选择成分。本解决方案中，前3个输出的成分能保留大约30%的原始数据信息。
tsvd.explained_variance_ratio_[0:3].sum()

# 可以创建一个运行TSVD的函数，使这个过程自动化(将参数 n_components 设置为原始特征数量-1).然后计算能够保留所需信息量的特征数量

tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)    # 用比原始特征数量小1的值作为n_components,创建并运行TSVD
features_tsvd = tsvd.fit(features)

tsvd_var_ratios = tsvd.explained_variance_ratio_    # 获取方差百分比数组


def select_n_components(var_ratio, goal_var):   # 函数定义
    total_variance = 0.0    # 设置总方差的初始值
    n_components = 0        # 设置特征数量的初始值
    for explained_variance in var_ratio:    # 遍历方差百分比数组的元素
        total_variance += explained_variance    # 将该百分比加入总方差
        n_components +=1    # n_components 的值加1
        if total_variance >= goal_var:  # 如果达到目标阈值
            break           # 结束遍历
    return n_components     # 返回n_components值


select_n_components(tsvd_var_ratios, 0.95)  # 运行函数
