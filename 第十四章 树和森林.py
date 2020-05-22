# 14.0 简介
# 基于树的学习算法是流行且应用广泛的一类非参数化的有监督学习算法。既可用于分类又可用于回归。
# 基于树的算法基础是包括一系列决策规则的决策树。
# 决策树型模型普及的一个原因是它们的解释性很强。可以通过绘制完整的决策树创建一个很直观的模型
# 从这个树系统可以引出各种各样的扩展，例如随机森林和堆叠(stacking)模型。

# 14.1 训练决策树分类器
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data
target = iris.target

decisiontree = DecisionTreeClassifier(random_state=0)   # 创建决策树分类器对象
model = decisiontree.fit(features, target)              # 训练模型
# 决策树的训练器会尝试找到在节点上能够最大限度降低数据不纯度(impurity)的决策规则。度量不纯度的方式很多。DecisionTreeClassifier 默认
# 使用基尼不纯度(Gini impurity):       G(t)=1-∑(c,i=1)pi^2     G(t)是节点t的基尼不纯度，pi是在节点t上的第i类样本的比例。
# 寻找使不纯度降低的决策规则的过程会被递归执行，直到所有叶子节点都变成纯节点(即仅包括一种分类)或达到某个终止条件。

observation = [[5, 4, 3, 2]]    # 创建新样本
model.predict(observation)      # 预测样本分类

model.predict_proba(observation)    # 查看样本分别属于三个分类的概率

decisiontree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0)      # 使用entropy作为不纯度检测方法创建决策树分类器对象
model_entropy = decisiontree_entropy.fit(features, target)      # 训练模型

# 14.2 训练决策树回归模型
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
boston = datasets.load_boston()
features = boston.data[:, 0:2]
target = boston.target

decisiontree = DecisionTreeRegressor(random_state=0)    # 创建决策树回归模型对象
model = decisiontree.fit(features, target)              # 训练模型
# 决策树回归模型工作方式与分类模型类似。不过前者不会使用entropy或者Gini impurity 的概念，
# 而是默认使用均方误差(MSE)的减少量来作为分裂规则的评估标准:  MSE = 1/n * ∑(n,i=1)(yi-yhat)^2

observation = [[0.02, 16]]      # 创建新样本
model.predict(observation)      # 预测样本值
# 可以用criterion参数来选择 分裂质量(split quality)的度量方式。例如可以用 平均绝对误差(MAE) 来构造决策树模型
decisiontree_mae = DecisionTreeRegressor(criterion="mae", random_state=0)   # 使用MAE 创建决策树回归模型
model_mae = decisiontree_mae.fit(features, target)                           # 训练模型

# 14.3 可视化决策树模型
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree
iris = datasets.load_iris()
features = iris.data
target = iris.target

decisiontree = DecisionTreeClassifier(random_state=0)
decisiontree.fit(features, target)

dot_data = tree.export_graphviz(decisiontree,
                                out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names)      # 创建DOT数据

graph = pydotplus.graph_from_dot_data(dot_data)     # 绘制图形
Image(graph.create_png())       # 显示图形
# 可以将整个模型可视化，是决策树分类器的优点之一。这也使得决策树成为机器学习中解释性最好的模型之一。
# 本解决方案中，模型以DOT格式(一种图形描述语言)导出，然后被绘制成图形。

graph.write_pdf("iris.pdf")     # 创建PDF
graph.write_png("iris.png")     # 创建PNG

# 14.4 训练随机森林分类器
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data
target = iris.target

randomforest = RandomForestClassifier(random_state=0, n_jobs=1)    # 创建随机森林分类器对象
model = randomforest.fit(features, target)                          # 训练模型

observation = [[5, 4, 3, 2]]    # 创建新样本
model.predict(observation)      # 预测样本的分类
# 决策树有个常见的问题,即倾向于紧密地拟合训练数据(过拟合)。这使得随机森林这种集成学习方法被普遍应用。在随机森林中, 许多决策树树同时被训练，
# 但每棵树只接收一个自举的(bootstrapped)样本(即有放回的随机抽样,抽样次数与原始样本数相同),并且每个节点在确定最佳分裂时只考虑全部特征的
# 一个子集。这个由随机决策树组成的森林(随机森林因此得名)通过投票决定样本的预测分类。

# RandomForestClassifier 使用与 DecisionTreeClassifier相同的参数。例如我们可以改变度量分裂质量的方法:
randomforest_entropy = RandomForestClassifier(
    criterion="entropy", random_state=0)           # 使用熵创建随机森林分类器对象
model_entropy = randomforest_entropy.fit(features, target)      # 训练模型
# 不过作为一个森林而不是一颗单独的决策树，RandomForestClassifier 有一些独特且重要的参数。
# max_features 决定每个节点需要考虑的特征的最大数量,允许输入的变量类型包括整数(特征的数量),浮点型(特征的百分比),sqrt(特征数量的平方根)
# 默认情况下设置为auto(相当于sqrt)。其次参数 bootstrap 用于设置在创建树时使用的样本子集，是有放回的抽样(默认值)还是无放回抽样。
# 第三参数n_estimators 设置森林中包括的决策树数量。最后n_jobs=-1 指定使用所有可用的CPU核进行训练。

# 14.5 训练随机森林回归模型
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
boston = datasets.load_boston()
features = boston.data[:, 0:2]
target = boston.target

randomforest = RandomForestRegressor(random_state=0, n_jobs=1)     # 创建随机森林回归对象
model = randomforest.fit(features, target)                          # 训练模型

# 随机森林回归模型几个重要参数:
# max_features 设置每个节点要考虑的特征的最大数量，默认为根号P个，其中P是特征的总数。
# bootstrap 设置是否使用有放回的抽样，默认值为True
# n_estimators 设置决策树的数量，默认为10

# 14.6 识别随机森林中的重要特征
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data
target = iris.target

randomforest = RandomForestClassifier(random_state=0, n_jobs=1)    # 创建随机森林分类器对象
model = randomforest.fit(features, target)      # 训练模型

importances = model.feature_importances_        # 计算特征的重要性

indices = np.argsort(importances)[::-1]         # 将特征的重要性按降序排列

names = [iris.feature_names[i] for i in indices]   # 按照特征的重要性对特征名称重新排序

plt.figure()        # 创建图
plt.title("Feature Importance")             # 创建图标题
plt.bar(range(features.shape[1]), importances[indices])      # 添加数据条
plt.xticks(range(features.shape[1]), names, rotation=90)      # 将特征名称添加为x轴标签
plt.show()

# 决策树可解释性是它的优点之一，对决策树模型可视化很容易。
# 但是一个随机森林模型由数十或数百棵决策树组成，很难对随机森林模型生成简易直观的可视化结果。
# 不过，我们可以用另一种方式可视化随机森林:比较(和可视化)每个特征的相对重要性。

# 关于特征的重要性有两点需要注意，首先scikit-learn需要将nominal型分类特征分解为多个二元特征，使得特征的重要性分散到各个二元特征中。
# 这样的话，即使原来的分类特征非常重要，分解后的特征往往也就没这么重要了。其次，如果两个特征高度相关，并且其中一个有很高的重要性，就会使
# 另一个特征的重要性显得稍低，如果不考虑这种情况，模型的效果会受到影响。   决策树和随机森林的分类及回归模型都可以通过feature_importances_
# 查看模型中每个特征的重要程度:
model.feature_importances_          # 查看特征的重要程度     数值越大，说明该特征越重要(所有特征的重要性分数相加等于1)

# 14.7 选择随机森林中的重要特征
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel
iris = datasets.load_iris()
features = iris.data
target = iris.target

randomforest = RandomForestClassifier(random_state=0, n_jobs=1)     # 创建随机森林分类器

selector = SelectFromModel(randomforest, threshold=0.3)             # 创建对象,选择重要性大于或等于阈值的特征

features_important = selector.fit_transform(features, target)       # 使用选择器创建的新特征矩阵

model = randomforest.fit(features_important, target)                # 使用重要的特征训练随机森林模型
# 在某些情况下，你可能希望减少模型中特征的数量。例如想减少模型的方差或者希望仅使用少数重要的特征来提高模型的可解释性。
# 在scikit-learn中，可以使用简单的两步工作流来创建一个使用较少特征的模型。首先，使用所有特征训练一个随机森林模型，并使用
# 训练得到的模型来确定重要的特征。接下来创建一个仅包括这些重要特征的新特征矩阵。本节中使用SelectFromModel方法来创建特征矩阵，
# 其中仅包括重要性大于或等于某阈值的特征。最后，使用这些特征创建一个新模型。
# 注意两点：1）经过one-hot 编码的 nominal型分类特征的重要性被稀释到二元特征中；2）一对高度相关的特征，其重要性被集中在其中一个特征上，
# 而不是均匀分布在这两个特征上。

# 14.8 处理不均衡的分类
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data
target = iris.target

features = features[40:, :]         # 删除前40个样本以获得高度不均匀的数据
target = target[40:]

target = np.where((target == 0), 0, 1)      # 创建目标向量表面分类是0还是1

randomforest = RandomForestClassifier(
    random_state=0, n_jobs=1, class_weight="balanced"
)                                           # 创建随机森林分类器对象

model = randomforest.fit(features, target)  # 训练模型
# 现实中用学习算法训练模型时，很容易遇到不均匀的分类问题。如果不解决这个问题会降低模型的性能。不过在scikit-learn中，很多学习算法都带有用于
# 纠正不均衡分类的内置方法。RandomForestClassifier 中的 class_weight 参数可以纠正不均匀分类的问题。如果将分类名和所需权重以字典的形式提供
# 如{"male":0.2, "female":0.8}, RandomForestClassifier将为各个分类相应地加权。不过参数balanced 通常更有用，它根据各个分类在数据中
# 出现的频率的倒数自动计算权重值:  wj = n/(knj)    wj是分类j的权重，n是样本数量，nj是分类j中的样本数量，k是分类的总数。
# balanced 可以增加较小分类的权重(减少较大分类的权重)

# 14.9 控制决策树的规模
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建决策树分类器对象
decisiontree = DecisionTreeClassifier(random_state=0,
                                      max_depth=None,           # 树的最大深度。如果该参数为0，则树一直生长直到所有叶子都为纯节点。如果提供整数，这棵树会被有效"修剪"到这个整数值表示的深度
                                      min_samples_split=2,      # 在该节点分裂之前,节点上最小的样本数。如果提供给整数，代表最小的样本数。如果提供浮点数,则最小样本数为总样本数乘以该浮点数
                                      min_samples_leaf=1,       # 叶子节点需要的最小样本数，与min_sample_split 使用相同的参数格式
                                      min_weight_fraction_leaf=0,
                                      max_leaf_nodes=None,      # 最大叶子节点数
                                      min_impurity_decrease=0   # 执行分裂所需的最小不纯度减少量
                                      )
# 一般情况下，只会用到 max_depth 和 min_impurity_split 这两个参数，因为较浅的树(有时候称为树桩stump)结构更简单，且方差较小

# 14.10 通过boosting 提高性能
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data
target = iris.target

adaboost = AdaBoostClassifier(random_state=0)       # 创建adaboost数分类器对象
model = adaboost.fit(features, target)              # 训练模型
# 在随机森林中，用一组随机决策树对目标向量进行预测。而一种替代的且更强大的学习方法称为boosting。有一种形式的boosting算法叫做AdaBoost，
# 它迭代地训练一系列弱模型(通常为浅层决策树，有时称为树桩),每次迭代都会为前一个模型预测错的样本分配更大的权重。AdaBoost算法的具体步骤如下:

# 1. 为每个样本xi 分配初始权重值 wi = 1/n，其中n是数据中样本的总数
# 2. 用数据训练弱模型
# 3. 对于每一个样本:
#   （1）如果弱模型对xi的预测是正确的，则wi减少。
#   （2）如果弱模型对xi的预测是错误的，则wi增大。
# 4. 训练一个新的弱模型，样本拥有更大的wi，优先级更高。
# 5. 重复步骤3和4，直到拥有更大的wi，优先级更高。

# 最终的结果是一个组合模型，里面不同的弱模型聚焦于(从预测角度来看) 更复杂的样本。 在scikit-learn中，我们使用AdaBoostClassifier或
# AdaBoostRegressor 实现 AdaBoost。其中最重要的参数是base_estimator、n_estimators和learning_rate
# base_estimator 表示训练弱模型的学习算法。这个参数几乎不需要改变，因为目前决策树(默认值)是AdaBoost最常用的学习算法。
# n_estimators 是需要迭代训练的模型数量
# learning_rate 是每个弱模型的权重变化率，默认值为1。减少这个参数值意味着权重幅度变小,这会使模型的训练速度变慢(但有时会使模型的性能更好)
# loss 是 AdaBoostRegressor 独有的参数，它设置了在更新权重时所用的损失函数。其默认值为线性损失函数还是可以改成平方(square)或指数函数(exponential)

# 14.11 使用袋外误差(Out-of-Bag Error)评估随机森林模型
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data
target = iris.target

randomforest = RandomForestClassifier(
    random_state=0, n_estimators=1000, oob_score=True, n_jobs=1)       # 创建随机数分析器对象

model = randomforest.fit(features, target)      # 训练模型

randomforest.oob_score_

# 在随机森林中,每个决策树使用自举的样本子集进行训练。这意味着对于每棵树而言，都有未参与训练的样本子集。这些样本被称为袋外(Out-of-Bag,OOB)样本。
# 袋外样本可以用作测试集来评估随机森林的性能。
# 对于每个样本，算法将其真实值与未使用该样本进行训练的树模型子集产生的预测值进行比较。计算所有样本的总得分，就能得到一个随机森林的性能指标。OOB分数
# 评估法可以用作交叉验证的替代方案。
# 在RandomForestClassifier 中 设置参数obb_score=True 可以计算obb分数，该分数可以使用 oob_score_来获取。
