# 13.1 拟合一条直线
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
boston = load_boston()
features = boston.data[:, 0:2]
target = boston.target
regression = LinearRegression()     # 创建线性回归对象
model = regression.fit(features, target)            # 拟合线性回归
# 线性回归假设特征与目标向量之间近似线性的关系。也就是说，特征对目标向量的影响(也称系数，权重或参数)是恒定的。
# yhat = β0+β1x1+β2x2+ε

model.intercept_                    # 查看截距
model.coef_                         # 查看特征权重
target[0] * 1000                    # 目标向量的第一个值乘以1000
model.predict(features)[0]*1000     # 预测第一个样本的目标向量并乘以1000
# 线性回归的优点是可解释性，模型的系数代表特征每单位变化对目标向量的影响
model.coef_[0]*1000                 # 第一个系数乘以1000

# 12.2 处理特征之间的影响
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
boston = load_boston()
features = boston.data[:, 0:2]
target = boston.target

interaction = PolynomialFeatures(
    degree=3, include_bias=False, interaction_only=True)        # 创建交互特征
features_interaction = interaction.fit_transform(features)

regression = LinearRegression()     # 创建线性回归对象
model = regression.fit(features_interaction, target)            # 拟合线性回归
# 有时候某个特征对目标向量的影响取决于另一个特征。将存在相互作用的一组特征的乘积作为新特征加入模型中，
# 就可以对这种特征进行建模，其公式为 yhat = β0+β1x1+β2x2+β3x1x2+ε        x1x2代表两者之间的相互作用
features[0]             # 查看第一个样本的特征
import numpy as np
interaction_term = np.multiply(features[:, 0], features[:, 1])  # 将每个样本的第一个和第二个特征相乘
interaction_term[0]     # 查看第一个样本的交互特征

# 虽然我们经常有理由相信两个特征之间存在相互作用，但有时并不是这样。 这种情况下，使用scikit-learn 的 PolynomialFeatures 为所有特征组合
# 创建交互项(交互特征)会很有用。 然后可以使用模型选择策略找出能产生最佳模型的特征和交互项组合。
# PolynomialFeatures 创建交互特征，必须设置3个重要的参数。
# interaction_only: 设置interaction_only=True 会告诉PolynomialFeature只返回交互特征(而不是多项式特征)
# 默认情况下 PolynomialFeatures 会添加"偏差"，可以通过 include_bias = False 来阻止加入偏差
# Degree: 用于确定最多用几个特征来创建交互特征
features_interaction[0]     # 观察

# 13.3 拟合非线性关系
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
boston = load_boston()
features = boston.data[:, 0:1]
target = boston.target

polynomial = PolynomialFeatures(degree=3, include_bias=False)       # 创建多项式特征x^2, x^3
features_polynomial = polynomial.fit_transform(features)

regression = LinearRegression()
model = regression.fit(features_polynomial, target)     # 拟合线性回归模型
# 多项式回归是线性回归的一种扩展，它使我们可以对非线性关系进行建模。
# 为什么可以将线性回归用于非线性函数？    因为这里只是为模型添加了特征，并没有改变线性回归拟合模型的方式。
features[0]
features[0]**2
features[0]**3
features_polynomial[0]      # x,x^2,x^3
# PolynomialFeatures 重要的参数
# degree:确定多项式特征的最高阶数
# include_bias: 默认情况下PolynomialFeatures 包括一个全为1的特征(称为偏差，bias)，可以设置include_bias=False来删除它

# 13.4 通过正则化减少方差
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
boston = load_boston()
features = boston.data
target = boston.target

scaler = StandardScaler()       # 特征标准化
features_standardized = scaler.fit_transform(features)

regression = Ridge(alpha=0.5)       # 创建一个包括指定alpha值的岭回归

model = regression.fit(features_standardized, target)   # 拟合线性回归模型

# 在标准线性回归中，通过最小化真实值和预测值的平方误差来训练模型，这个平方误差值也称残差平方和(RSS, Residual Sum of Squares)
# RSS = ∑(n,i=1)(yi-yihat)
# 正则化的回归模型与其相似，但是它在优化对象RSS上加入了对系数值的惩罚, 该惩罚项被称为收缩惩罚(shrinkage penalty), 因为它试图"缩小"模型。
# 常见的正则化优化方法:岭回归和套索回归。 二者的差异在于所使用的的收缩惩罚项不同。
# 岭回归中，收缩惩罚项是可调参数与所有系数的平方和的乘积:  RSS+α∑(p,j=1)βj^2
# α是超参数。βj是 总计p个特征中第j个特征的系数

# 套索回归与其类似,只不过收缩惩罚项变成了可调超参数与所有系数绝对值之和的乘积:   (1/2n)RSS+α∑(p,j=1)|βj|
# n是样本数量。

# 岭回归通常比套索回归产生的结果稍好, 但是套索回归产生的模型更容易解释。
# 如果想要在岭回归和套索回归的惩罚项之间折中，可以使用弹性网络(elastic net) 包含两者。
# 岭回归和套索回归都可以通过损失函数中加入βj来惩罚大型或复杂模型。超参数α用来控制βj的惩罚强度，α越大，生成的模型越简单。
# α的理想值应该是像其他超参数一样通过调试获得的。  scikit-learn 包含一个RidgeCV方法，可以使用它来选择理想值
from sklearn.linear_model import RidgeCV
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])     # 创建包含三个alphas值的RidgeCV对象
model_cv = regr_cv.fit(features_standardized, target)       # 拟合线性回归
model_cv.coef_      # 查看模型系数
model_cv.alpha_     # 查看最优模型的alpha值
# 在线性回归中系数的值受特征的范围（scale）的影响，而在正则化模型中所有系数会被加在一起,所以在训练模型之前必须确保特征已经标准化<前提>


# 13.5 使用套索回归减少特征
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
boston = load_boston()
features = boston.data
target = boston.target

scaler = StandardScaler()           # 特征标准化
features_standardized = scaler.fit_transform(features)

regression = Lasso(alpha=0.5)       # 创建套索回归，并指定alpha值
model = regression.fit(features_standardized, target)   # 拟合线性回归
# 套索回归的惩罚项可以将特征的系数减少为0,从而有效地减少模型中特征的数量。
model.coef_     # 查看系数      为0的系数意味着它们对应的特征并未在模型中使用

regression_a10 = Lasso(alpha=10)    # 创建一个alpha值为10的套索回归
model_a10 = regression_a10.fit(features_standardized, target)
model_a10.coef_
# 利用这种特性，可以在特征向量中包含100个特征，然后调整套索回归的超参数，生成比如仅使用10个最重要的特征的模型。
# 这样做可以减少模型方差，同时提高模型的可解释性。(特征越少就越容易解释)

