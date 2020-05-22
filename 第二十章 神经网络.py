# 20.0 简介
# 神经网络的核心是神经元(unit).神经元接收了一个或者多个输入,为每个输入乘以一个参数(权重),接着对加权之后的输入值求和再加上某个偏差值(一般是1)
# ,最后把这个值反馈给一个激活函数。这个输出值会被向前传递给神经网络更深层的神经元(如果还有神经元的话)。

# 前馈(feedforward)神经网络,又叫做多层感知器(multilayer perception),是现实世界中最简单的人工神经网络。神经网络可以视为由一系列相互连接
# 的层组成的网络,它的一端连接着观察值的一个特征值,另一端连接着对应的目标值(比如观察值的分类)。观察值的特征值在网络中向前传播,每经过一层,网络
# 都会对特征值进行转换,目标是让最后的输出与目标值相同,'前馈'这个名字就源于此。

# 具体来说,前馈神经网络包括3种类型的神经元层。神经网络的起始端有一个输入层,输入层的每一个神经元包含一个观察值的某一个特征值。比如,如果观察值有
# 100个特征,那么输入层就有100个节点。神经网络的末端有一个输出层,它把隐藏层的输出转换成对我们的任务有用的值。有多个隐藏层的神经网络(比如10,
# 100,1000层)被认为是很"深"的网络。它们的应用又被称为"深度学习"(deep learning)。

# 一般而言,神经网络在创建时所有的参数都会被初始化为符合高斯分布或者正态分布的小随机值。一旦一个观察值(更常见的情况是一组观察值的集合,又被称为
# 批次<batch>)被传递给神经网络,它的输出值就会被拿来用损失函数与观察值的真实值进行比较。这个过程叫前向传播(forward propagation)。接下来,
# 算法"向后"在神经网络中传播,它识别每个参数对预测值和真实值之间的差异的影响程度,这个过程叫做反向传播(backward propagation)。对于每一个参数,
# 优化算法决定应该如何调整权重值才能改善输出值。 神经网络通过训练集中的每个观察值重复做多次向前传播和反向传播(所有观察值都通过网络传递一次
# 就被称为一个epoch,这种训练一般都需要多个epoch),以此迭代更新输出值。

# 本章中使用Keras来创建、训练和评估几种神经网络。Keras是一个高层软件包,使用了像TensorFlow和Theano这样的库来作为它的"引擎"。对于我们来说,
# Keras的优势在于它可以使我们专注于神经网络的设计和训练,把张量运算的细节留给其他库来完成。
# 使用Keras创建的神经网络既可以用CPU训练,也可以用GPU(专门为深度学习而设计的计算机)训练。现实世界中,我们强烈建议使用GPU来训练神经网络;但
# 为了便于学习,本书中所有的神经网络都很小而且很简单,用笔记本电脑也可以在几分钟之内把它们训练出来。对于大型神经网络和大量训练数据,使用CPU来训练
# 会比适用GPU慢得多

# 20.1 为神经网络预处理数据
from sklearn import preprocessing
import numpy as np
features = np.array([[-100.1, 3240.1],
                    [-200.2, -234.1],
                    [5000.5, 150.1],
                    [6000.6, -125.1],
                    [9000.9, -673.1]])       # 创建特征
scaler = preprocessing.StandardScaler()     # 创建scaler
features_standardized = scaler.fit_transform(features)  # 转换特征
features_standardized
# 一般来说,一个神经网络的参数会被初始化(或者说被创建)为一些小的随机数。如果特征值比参数大很多,神经网络往往表现得不如人意。另外观察值的特征值
# 经过这些神经元的传递后,会进行相加,所以让所有特征值拥有相同的单位就很重要。
# 由于上面这些原因,最佳实践(尽管有些时候并非必须如此,比如当我们的数据都是二元特征的时候)就是先标准化每一个特征值使其均值为0,标准差为1。
print("Mean:", round(features_standardized[:, 0].mean()))
print("Standard deviation", features_standardized[:, 0].std())

# 20.2 设计一个神经网络
from keras import models
from keras import layers
network = models.Sequential()   # 启动神经网络

network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))   # 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16, activation="relu"))                      # 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16, activation="sigmoid"))                   # 添加使用sigmoid激活函数的全连接层

# 编译神经网络
network.compile(loss="binary_crossentropy",         # 交叉熵
                optimizer="rmsprop",                 # 均方根传播
                metrics=["accuracy"])               # 将准确率作为性能指标
# 神经网络是由多层神经元组成的,神经元层的类型以及它们组成神经网络的方式非常多。现阶段景观有很多被广泛使用的架构模式(我们会在本章中讲到),但
# 真相是,选择正确的架构更像是一门艺术,需要我们做很多研究工作。
# 要在Keras中构建一个前馈神经网络,我们需要对网络架构和训练过程做许多选择。记住,隐藏层的每个神经元都会经历如下步骤:
# 1.接收一些输入
# 2.给每个输入乘以一个参数作为权重
# 3.对所有加权过的输入求和,再加上偏差(一般是1)
# 4.接下来把这个值应用到某个函数上(又叫做激活函数)
# 5.把输出传递给下一层的神经元

# 第一:对隐藏层和输入层中定的每一层,我们必须义神经元的数量和它的激活函数。总的来说,一个层中神经元越多,神经网络就越能学习复杂的模式。尽管如此,
# 神经元过多可能会使神经网络对训练数据过拟合,影响其在测试数据上的表现。
# 对隐藏层来说,一个流行的激活函数是矫正的线性单元(Rectified Linear Unit,ReLU):f(z) = max(0,z)
# z是加权过的输入和偏差之和。我们可以看到如果z大于0,激活函数就会返回z值否则返回0.这个简单的激活函数有许多可贵的特性(本书不讨论这些特性),使
# 它成为神经网络中受欢迎的激活函数。
# 第二:我们需要决定神经网络中隐藏层的数量。层数越多,神经网络能学习的关系就越复杂,但计算开销也会越大。
# 第三:我们必须决定输出层的激活函数(如果有的话)的结构。输出函数的本质经常由神经网络的目标决定,这里列出一些常见的输出层的模式:
#   二元分类:一个有sigmoid激活函数的神经元
#   多元分类:k个神经元(这里的k是目标分类的个数)和一个softmax激活函数
#   回归:一个没有激活函数的神经元
# 第四:我们需要定义一个损失函数(用来衡量预测值和真实值的符合程度)。这个函数也经常是由问题的类型所决定的:
#   二元分类: 二分交叉熵 (Binary Cross-entropy)
#   多元分类: 分类交叉熵 (Categorical cross-entropy)
#   回归:均分误差
# 第五:我们需要定义一个优化器,它可以被直观地理解为我们的策略"绕过了"损失函数,并且找到了产生最小误差的那些参考值。常见的优化器有:随机梯度下
# 降、动量随机梯度下降、均方根传播和自适应矩估计(关于这些优化器的更多信息可见 https://arxiv.org/abs/1702.05659)
# 第六:我们可以选择一个或者多规格指标来评估神经网络的性能,比如准确率

# Keras提供两种创建神经网络的方法。Keras的sequential 模型可以把神经元层堆叠起来以创建神经网络。另一种创建神经网络的方法叫做函数API,这种
# 方法更适合研究人员使用而不是商用

# 解决方案中,我们使用Keras的sequential模型创建了一个两层的神经网络(计算层数时我们不会把输入层算在内,因为它没有任何参数需要学习)。每一层都
# 是"紧密的"(又叫做全连接的),这意味着前一层的所有神经元都和下一层的所有神经元相连。在第一个隐藏层中,我们设定units=16,表示这一层有16个神经元,
# 每个神经元都有ReLu激活函数(activation='relu')。在Keras中，任何神经网络的第一个隐藏层必须包括一个input_shape参数,它表示特征数据的形状。
# 比如（10,）就告诉我们,第一层期望每个观察值有10个特征值。第二层和第一层一样,只不过不需要加上input_shape参数。我们的神经网络是设计来做二元
# 分类的,所以输出层仅包括一个带sigmoid激活函数的神经元,它将输出限制在0和1之间(表示一个观察值属于分类1的概率)。最后在训练模型之前,还需要告诉
# Keras,我们想让网络如何学习。使用compile方法,加上优化算法(RMSProp)、损失函数(binary_crossentropy),以及一个或者多个性能衡量标准来告诉
# Keras如何学习。


# 20.3 训练一个二元分类器
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
np.random.seed(0)       # 设定随机种子
number_of_features = 1000   # 设定想要的特征数量
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)  # 从影片加载数据和目标向量

tokenizer = Tokenizer(num_words=number_of_features)         # 将影评数据转化为one-hot 编码过的特征矩阵
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()       # 创建神经网络对象

network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))   # 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16, activation="relu"))                      # 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=1, activation="sigmoid"))                   # 添加使用sigmoid激活函数的全连接层

# 编译神经网络
network.compile(loss="binary_crossentropy",         # 交叉熵
                optimizer="rmsprop",                 # 均方根传播
                metrics=["accuracy"])               # 将准确率作为性能指标

# 训练神经网络                    训练神经网络需要指定6个重要的参数
history = network.fit(features_train,               # 特征
                      target_train,                 # 目标向量
                      epochs=3,                     # epoch数量
                      verbose=1,                   # 每个epoch之后打印描述
                      batch_size=100,               # 每个批次中观察值的数量
                      validation_data=(features_test, target_test))     # 测试数据
# epochs参数指定训练数据时需要使用多少个epoch,verbose参数决定训练过程中需要输出多少信息:0表示没有输出,1表示输出一个进度条,2表示每个epoch输出一条日志
# batch_size 设定在计算多少个观察值之后才更新参数
# 最后使用validation_data参数来评估模型,可以放测试用的特征和目标向量 或者 使用validation_split 来设定要留多少比例的训练集数据用于评估模型
features_train.shape         # 查看特征矩阵的形状

# 我们用了50,000 条影评数据 (25,000用作训练数据 25,000用作测试数据)然后把他们分为了正面评价和负面评价。我们把评论的文本转换为1000个二元
# 特征值来表示1000个最常出现的词出现与否。简单来说,我们的神经网络会使用25,000个观察值来预测一条影评是正面还是负面的评价。其中观察值有1000个特征。


# 20.4 训练一个多元分类器
import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
np.random.seed(0)
number_of_features = 5000       # 设定我们想要的特征的数量

data = reuters.load_data(num_words=number_of_features)
(data_train, target_vector_train), (data_test, target_vector_test) = data       # 加载特征和目标数据

tokenizer = Tokenizer(num_words=number_of_features)         # 把特征数据转化为one-hot 编码过的特征矩阵
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

target_train = to_categorical(target_vector_train)          # 把one-hot编码的特征向量转换成特征矩阵
target_test = to_categorical(target_vector_test)

network = models.Sequential()                               # 启动神经网络

network.add(layers.Dense(units=100, activation="relu", input_shape=(number_of_features,)))   # 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=46, activation="softmax"))                      # 添加使用softmax激活函数的全连接层

# 编译神经网络
network.compile(loss="categorical_crossentropy",         # 交叉熵
                optimizer="rmsprop",                 # 均方根传播
                metrics=["accuracy"])               # 将准确率作为性能指标

# 训练神经网络                    训练神经网络需要指定6个重要的参数
history = network.fit(features_train,               # 特征
                      target_train,                 # 目标向量
                      epochs=3,                     # epoch数量
                      verbose=0,                   # 没有输出
                      batch_size=100,               # 每个批次中观察值的数量
                      validation_data=(features_test, target_test))     # 测试数据

# 这次数据是11,228个路透社新闻专栏。每个新闻专栏被分成46个主题,我们把新闻专栏转换为5000个二元特征值(分别表示在该专栏中这5000个词中的某个词
# 是否出现过)来作为特征数据。我们用one-hot编码将目标数据转换成一个目标矩阵,每一行表示一个观察值属于46个分类中的哪一个
target_train        # 查看目标矩阵
# 第二,我们增加了隐藏层的神经元数量,帮忙神经网络表示46个分类之间更复杂的关系。
# 第三,因为这是一个多元分类的问题,我们使用了有46个神经元(每个神经元对应一个分类)的输出层,其中包含一个softmax激活函数。这个激活函数会返回一个
# 有46个值的矩阵,这46个值之和为1。它们表示一个观察值被归类成46个分类之一的概率。
# 第四,我们使用了一个适合多元分类问题的损失函数,即分类交叉熵损失函数 categorical_crossentropy

# 20.5 训练一个回归模型
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
np.random.seed(0)
features, target = make_regression(n_samples=10000,
                                   n_features=3,
                                   n_informative=3,
                                   n_targets=1,
                                   noise=0.0,
                                   random_state=0)          # 生成特征矩阵和目标向量

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.33, random_state=0)   # 把数据分为训练集和测试集

network = models.Sequential()       # 启动神经网络
network.add(layers.Dense(units=32,
                         activation='relu',
                         input_shape=(features_train.shape[1],)))
network.add(layers.Dense(units=32, activation='relu'))
network.add(layers.Dense(units=1))      # 添加没有激活函数的全连接层

# 编译神经网络
network.compile(loss="mse",             # 均方误差
                optimizer="RMSprop",    # 优化算法
                metrics=["mse"])        # 均方误差

# 训练神经网络
history = network.fit(features_train,
                      target_train,
                      epochs=10,
                      verbose=0,
                      batch_size=100,                                       # 每个批次的观察值数量
                      validation_data=(features_test, target_test))         # 测试数据
# 用神经网络预测连续的数值是完全可以做到的。因为我们训练的是回归模型,所以应该使用合适的损失函数和性能评估指标。例子中使用的是MSE
# MSE= 1/n * ∑(n,i=1)(yhat-yi)^2    注意: 因为我们使用 make_regression 产生模拟数据,不需要标准化数据,实际情况中需要对数据进行标准化处理

# 20.6 做预测
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
np.random.seed(0)
number_of_features = 10000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             # 把IMDB 数据转换为one-hot编码的特征矩阵
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   # 启动神经网络

network.add(layers.Dense(units=16,
                         activation='relu',
                         input_shape=(number_of_features,)))
network.add(layers.Dense(units=16, activation='relu'))
network.add(layers.Dense(units=1,activation="sigmoid"))

# 编译神经网络
network.compile(loss="binary_crossentropy",         # 交叉熵
                optimizer="rmsprop",                 # 均方根传播
                metrics=["accuracy"])               # 将准确率作为性能指标

# 训练神经网络                    训练神经网络需要指定6个重要的参数
history = network.fit(features_train,               # 特征
                      target_train,                 # 目标向量
                      epochs=3,                     # epoch数量
                      verbose=0,                   # 每个epoch之后打印描述
                      batch_size=100,               # 每个批次中观察值的数量
                      validation_data=(features_test, target_test))     # 测试数据

predicted_target = network.predict(features_test)       # 预测测试集的分类

predicted_target[0]         # 查看第一个观察值属于分类1的预测概率
# 观察值越接近0,则属于分类0。观察值越接近1,则属于分类1。


# 20.7 可视化训练历史
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
import matplotlib.pyplot as plt
np.random.seed(0)
number_of_features = 10000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             # 把IMDB 数据转换为one-hot编码的特征矩阵
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   # 启动神经网络

network.add(layers.Dense(units=16,
                         activation='relu',
                         input_shape=(number_of_features,)))
network.add(layers.Dense(units=16, activation='relu'))
network.add(layers.Dense(units=1, activation="sigmoid"))

# 编译神经网络
network.compile(loss="binary_crossentropy",         # 交叉熵
                optimizer="rmsprop",                 # 均方根传播
                metrics=["accuracy"])               # 将准确率作为性能指标

# 训练神经网络                    训练神经网络需要指定6个重要的参数
history = network.fit(features_train,               # 特征
                      target_train,                 # 目标向量
                      epochs=15,                     # epoch数量
                      verbose=0,                   # 每个epoch之后打印描述
                      batch_size=1000,               # 每个批次中观察值的数量
                      validation_data=(features_test, target_test))     # 测试数据

training_loss = history.history["loss"]         # 获取训练集和测试集的损失历史数值
test_loss = history.history["val_loss"]

epoch_count = range(1, len(training_loss)+1)    # 为每个epoch编号

plt.plot(epoch_count, training_loss, "r--")     # 画出历史损失数值
plt.plot(epoch_count, test_loss, "b-")
plt.legend(['Training Loss', "Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

training_accuracy = history.history["accuracy"]      # 获取训练集和测试集的准确率历史数值
test_accuracy = history.history["val_accuracy"]
plt.plot(epoch_count, training_accuracy, "r--")
plt.plot(epoch_count, test_accuracy, "b-")

plt.legend(["Training Accuracy", "Test Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Score")
plt.show()
# 一开始我们的神经网络是新的,性能会比较差。而随着神经网络从训练数据中学习,其在训练集和测试集上的误差都会逐渐降低。
# 但是,在某个时间点之后,神经网络开始"记住"训练数据,并且产生过拟合。当出现这种情况时,训练集误差会减小而测试集误差会增大。 因此,在很多情况下
# 会有一个"甜蜜点"(sweet spot),到达这个点时,测试集误差(这是我们主要关注的误差)最小。

# 20.8 通过权重调节减少过拟合
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import regularizers
np.random.seed(0)
number_of_features = 1000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             # 把IMDB 数据转换为one-hot编码的特征矩阵
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   # 启动神经网络

network.add(layers.Dense(units=16,
                         activation='relu',
                         kernel_regularizer=regularizers.l2(0.01),
                         input_shape=(number_of_features,)))
network.add(layers.Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
network.add(layers.Dense(units=1, activation="sigmoid"))

# 编译神经网络
network.compile(loss="binary_crossentropy",         # 交叉熵
                optimizer="rmsprop",                 # 均方根传播
                metrics=["accuracy"])               # 将准确率作为性能指标

# 训练神经网络                    训练神经网络需要指定6个重要的参数
history = network.fit(features_train,               # 特征
                      target_train,                 # 目标向量
                      epochs=3,                     # epoch数量
                      verbose=0,                   # 每个epoch之后打印描述
                      batch_size=100,               # 每个批次中观察值的数量
                      validation_data=(features_test, target_test))     # 测试数据
# 对抗过拟合神经网络的策略之一就是惩罚神经网络的参数(权重),使他们变得很小的值,从而创建一个更简单、更难过拟合的模型。这个方法叫做权重调节或者
# 权重减少(weight decay)更准确地说,权重调节就是将一个惩罚项加在L2范数这样的损失函数上。添加参数kernel_regularizer=regularizers.l2(0.01)
# 0.01表示要对参数值施加多重惩罚。

# 20.9 通过提前结束减少过拟合
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
np.random.seed(0)
number_of_features = 1000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             # 把IMDB 数据转换为one-hot编码的特征矩阵
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   # 启动神经网络

network.add(layers.Dense(units=16,
                         activation='relu',
                         input_shape=(number_of_features,)))
network.add(layers.Dense(units=16, activation='relu'))
network.add(layers.Dense(units=1, activation="sigmoid"))

# 编译神经网络
network.compile(loss="binary_crossentropy",         # 交叉熵
                optimizer="rmsprop",                 # 均方根传播
                metrics=["accuracy"])               # 将准确率作为性能指标

callbacks = [EarlyStopping(monitor="val_loss", patience=2),
             ModelCheckpoint(filepath="best_model.h5",
                             monitor="val_loss",
                             save_best_only=True)]      # 设置一个回调函数来提前结束训练,并保存训练结束时的最佳模型

# 训练神经网络                    训练神经网络需要指定6个重要的参数
history = network.fit(features_train,               # 特征
                      target_train,                 # 目标向量
                      epochs=20,                     # epoch数量
                      callbacks=callbacks,          # 提前结束
                      verbose=1,                   # 每个epoch之后打印描述
                      batch_size=100,               # 每个批次中观察值的数量
                      validation_data=(features_test, target_test))     # 测试数据
# 一个最普遍且最有效的对抗过拟合的方法是:监视训练过程并且在测试集误差开始增大时就结束训练。这个策略被称作提前结束。
# 我们用回调函数来实现提前结束策略。回调函数是可以再训练过程的某几个特定阶段应用的函数,比如在每个epoch结束的时候,加入EarlyStopping(monitor
# ='val_loss', patience=2) 告诉程序我们想要监视每个epoch的测试集(验证集)损失,并且如果连续两个epoch测试集损失的情况都没有得到改善,就会中断训练
# 但是因为我们设定了patience=2,所以得不到最佳模型。而会得到最佳模型之后的两个epoch的模型。因此我们也可以添加一个ModelCheckpoint操作,
# 在每个检查点(如果你有一个持续多日的训练由于某种原因被打断,这时检查点就会很有用)之后把模型保存到文件中。如果设定了save_best_only=True，
# ModelCheckpoint 就会仅保存最佳模型。

# 20.10 通过Dropout 减少过拟合
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
np.random.seed(0)
number_of_features = 1000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             # 把IMDB 数据转换为one-hot编码的特征矩阵
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   # 启动神经网络
network.add(layers.Dropout(0.2, input_shape=(number_of_features,)))       # 为输入层添加一个Dropout层
network.add(layers.Dense(units=16, activation="relu"))                  # 添加使用ReLU激活函数的全连接层
network.add(layers.Dropout(0.5))        # 为前面的隐藏层添加一个Dropout层
network.add(layers.Dense(units=16,activation="relu"))       # 添加使用ReLU激活函数的全连接层
network.add(layers.Dropout(0.5))        # 为前面的隐藏层添加一个Dropout层
network.add(layers.Dense(units=1, activation="sigmoid"))    # 添加使用sigmoid激活函数的全连接层

# 编译神经网络
network.compile(loss="binary_crossentropy",         # 交叉熵
                optimizer="rmsprop",                 # 均方根传播
                metrics=["accuracy"])               # 将准确率作为性能指标

# 训练神经网络                    训练神经网络需要指定6个重要的参数
history = network.fit(features_train,               # 特征
                      target_train,                 # 目标向量
                      epochs=3,                     # epoch数量
                      verbose=0,                   # 每个epoch之后打印描述
                      batch_size=100,               # 每个批次中观察值的数量
                      validation_data=(features_test, target_test))     # 测试数据
# Dropout是一个流行的、强大的调节神经网络的办法。在Dropout方法中,每创建一个批次的观察值用于训练时,一层或者多层的一部分神经元会被乘以0(即被丢弃)
# 虽然每个批次都在同一个网络中训练的(比如,有同样的参数),但是每个批次面对的网络结构都有些许差异。
# Dropout方法很有效是因为它不断随机地丢弃每个批次中的神经元,迫使神经元在各种网络结构下依然能够学习参数。换句话说,神经元们变得对其他隐藏的神经元的中断
# (也可以理解为噪音)更加健壮,这样可以阻止网络记住训练集数据。
# 在隐藏层和输入层都可以添加Dropout方法。当一个输入层被丢弃后,它的特征值就不会在那个批次中被传进网络。一般对神经元丢弃比例为,输入层0.2,隐藏层0.5
# 在Keras中,可以通过在网络构架中添加若干个Dropout层来实现Dropout方法。每个Dropout层会在每个批次中丢弃前一层传过来的用户定义数量的神经元,
# 这个数量是超参数。记住在Keras中输入层被认定为第一层,并且不需要用add方法来添加。所以,如果想给输入层添加Dropout方法,给网络结构加的第一层
# 就是Dropout层。这一层输入层神经元的丢弃比例0.2,也包括定义观察值的形状input_shape。接着,我们在每个隐藏层之后添加一个神经元丢弃比例为0.5的Dropout层。

# 20.11 保存模型训练过程
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint
np.random.seed(0)
number_of_features = 1000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             # 把IMDB 数据转换为one-hot编码的特征矩阵
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   # 启动神经网络
network.add(layers.Dense(units=16,activation="relu",input_shape=(number_of_features,)))       # 为输入层添加一个Dropout层
network.add(layers.Dense(units=16,activation="relu"))       # 添加使用ReLU激活函数的全连接
network.add(layers.Dense(units=1, activation="sigmoid"))    # 添加使用sigmoid激活函数的全连接层

# 编译神经网络
network.compile(loss="binary_crossentropy",         # 交叉熵
                optimizer="rmsprop",                 # 均方根传播
                metrics=["accuracy"])               # 将准确率作为性能指标

checkpoint = [ModelCheckpoint(filepath="model.hdf5")]

# 训练神经网络                    训练神经网络需要指定6个重要的参数
history = network.fit(features_train,               # 特征
                      target_train,                 # 目标向量
                      epochs=3,                     # epoch数量
                      callbacks=checkpoint,         # 检查点
                      verbose=0,                   # 每个epoch之后打印描述
                      batch_size=100,               # 每个批次中观察值的数量
                      validation_data=(features_test, target_test))     # 测试数据
# 使用ModelCheckPoint一个更实际的问题。在真实世界中,神经网络一般需要训练几个小时甚至几天。在这段时间内,有很多环节都可能出错:电脑可能没电,
# 服务器可能崩溃,或者某个冒失鬼关闭了你的笔记本电脑。
# ModelCheckPoint 会在每一个epoch之后保存模型,以避免这一类问题。具体来说,就是在每个epoch之后,ModelCheckPoint把模型保存到filepath参数
# 指定的路径中。如果只给定一个文件名比如解决方案中的models.hdf5,那么这个文件在每个epoch后都会被最近的模型重写。如果至香港根据某个损失函数的表现
# 来保存最佳模型,可以设置save_best_only=True 和 monitor='val_loss',这样如果现有模型比前一个模型的测试集损失更大的话,该文件并不会被重写。
# 还有一种方案,我们可以保存每个epoch的模型,单独作为一个文件,并且将epoch编号和测试集损失值写在文件名中，比如设置filepath参数为
# "model_{epoch:02d}_{val_loss:.2f}.hdf5" ,如果在第11个epoch之后保存,测试集损失值为0.33的话,包含这个模型的文件名字为 model_10_0.33.hdf5


# 20.12 使用k折交叉验证评估神经网络
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
np.random.seed(0)
number_of_features = 100
features, target = make_classification(n_samples=10000,
                                       n_features=number_of_features,
                                       n_informative=3,
                                       n_redundant=0,
                                       n_classes=2,
                                       weights=[.5, .5],
                                       random_state=0)


def create_network():   # 创建一个函数,返回编译过的网络
    network = models.Sequential()
    network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))
    network.add(layers.Dense(units=16, activation="relu"))
    network.add(layers.Dense(units=1, activation="sigmoid"))
    network.compile(loss="binary_crossentropy",
                    optimizer="rmsprop",
                    metrics=["accuracy"])
    return network


neural_network = KerasClassifier(build_fn=create_network,           # 封装Keras模型,以便它能被scikit-learn使用
                                 epochs=10,
                                 batch_size=100,
                                 verbose=0)

cross_val_score(neural_network, features, target, cv=3)     # 使用三折交叉验证来评估神经网络
# 神经网络经常用于非常大的数据集而且可能需要几小时甚至几天来训练,所以如果训练时间很长的话,采用k折交叉验证就会增加计算开销,这并不是一个值得推荐的做法。
# 常见的稳妥做法是用某个测试集评估神经网络。如果数据规模不大,k折交叉验证可以被用于最大化我们评估神经网络性能的能力。在Keras中,这是可行的,因为
# 我们可以封装所有的神经网络,使他们能使用scikit-learn中的评估特性,包括k折交叉验证。为了实现这一点,必须县创建一个函数,返回编译好的神经网络。
# 接着使用KerasClassifier(这里假设有一个分类器)来封装模型,使它可以被scikit-learn使用。然后神经网络就可以像其他scikit-learn的学习算法一样被使用了。

# 20.13 调校神经网络
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
np.random.seed(0)
number_of_features = 100
features, target = make_classification(n_samples=10000,
                                       n_features=number_of_features,
                                       n_informative=3,
                                       n_redundant=0,
                                       n_classes=2,
                                       weights=[.5, .5],
                                       random_state=0)


def create_network(optimizer="rmsprop"):   # 创建一个函数,返回编译过的网络
    network = models.Sequential()
    network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))
    network.add(layers.Dense(units=16, activation="relu"))
    network.add(layers.Dense(units=1, activation="sigmoid"))
    network.compile(loss="binary_crossentropy",
                    optimizer=optimizer,
                    metrics=["accuracy"])
    return network


neural_network = KerasClassifier(build_fn=create_network, verbose=0)           # 封装Keras模型,以便它能被scikit-learn使用

# 创建超参数
epochs = [5, 10]
batches = [5, 10, 100]
optimizers = ["rmsprop", "adam"]

hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)     # 创建超参数选项

grid = GridSearchCV(estimator=neural_network, param_grid=hyperparameters)           # 创建网格搜索

grid_result = grid.fit(features, target)

grid_result.best_params_        # 查看最优神经网络的超参数
# 如果你的模型需要12个小时或者一天的时间来训练,那么网格搜索可能需要花上一周甚至更长的时间。因此,神经网络自动超参数调校不是万能药,但是它在某些特定情形下是有用的。

# 20.14 可视化神经网络
from keras import models
from keras import layers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

network = models.Sequential()
network.add(layers.Dense(units=16, activation="relu", input_shape=(10, 1)))
network.add(layers.Dense(units=16, activation="relu"))
network.add(layers.Dense(units=1, activation="sigmoid"))

SVG(model_to_dot(network, show_shapes=True).create(prog="dot", format="svg"))    # 可视化网络结构


plot_model(network, show_shapes=True,to_file="network.png")                      # 将可视化后的网络结构图保存为文件

# Keras 提供了工具函数用于快速可视化神经网络。如果想在Jupyter Notebook 中显示一个神经网络,可以使用model_to_dot。show_shapes 参数指定
# 是否展示输入和输出的形状,它可以帮助我们调试网络。如果想展示一个更简单的模型,可以设置show_shapes=False
SVG(model_to_dot(network, show_shapes=False).create(prog="dot", format="svg"))

# 20.15 图像分类
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_data_format("channels_first")   # 设置色彩通道值优先
np.random.seed(0)

# 图像信息
channels = 1
height = 28
width = 28

(data_train, target_train), (data_test, target_test) = mnist.load_data()     # 从MNIST数据集中读取数据和目标
data_train = data_train.reshape(data_train.shape[0], channels, height, width)   # 将训练集图像数据转换成特征
data_test = data_test.reshape(data_test.shape[0], channels, height, width)    # 将测试集图像数据转换特征

features_train = data_train / 255   # 将像素强度值缩小到0到1之间
features_test = data_test / 255

target_train = np_utils.to_categorical(target_train)        # 将目标进行one-hot编码
target_test = np_utils.to_categorical(target_test)
number_of_classes = target_test.shape[1]

network = Sequential()

network.add(Conv2D(filters=64,
                   kernel_size=(5, 5),
                   input_shape=(channels, width, height),
                   activation="relu"))      # 添加有64个过滤器、一个大小为5X5的窗口和ReLu激活函数的卷积层
network.add(MaxPooling2D(pool_size=(2, 2)))     # 添加一个2X2窗口的最大池化层
network.add(Dropout(0.5))       # 添加Dropout层
network.add(Flatten())      # 添加一层来压平输入
network.add(Dense(128, activation="relu"))      # 添加带ReLU激活函数的有128个神经元的全连接层
network.add(Dropout(0.5))       # 添加Dropout层
network.add(Dense(number_of_classes, activation="softmax"))      # 使用softmax激活函数的全连接层

network.compile(loss="categorical_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])

network.fit(features_train,
            target_train,
            epochs=2,
            verbose=0,
            batch_size=1000,
            validation_data=(features_test, target_test))
# 卷积神经网络(Convolutional Neural Networks, aka. ConvNets) 是一种流行的神经网络,在计算机视觉领域非常有效。 对图像使用前馈神经网络是可行的
# 但是有两大问题:1.前馈神经网络没有考虑像素之间的空间结构。2.前馈神经网络学习的是特征的全局关系而不是局部的模式,这意味着前馈神经网络无法识别一个物体,
# 不管该物体出现在图像的何处。

# 卷积神经网络可以解决以上两个问题(还可以解决其他问题),一张图像的数据可以被想象成一个三维张量:宽度x高度x深度 (又叫特征图)
# 在卷积神经网络中,卷积可以被想象成在图像的像素上滑动一个窗口,通过这个窗口查看像素和它周围的邻居们。接着它把初始图像数据转换成一个新的三维张量
# 头两个张量近似宽度和高度,第三个维度(包括颜色值)现在表示像素"属于"哪种模式(eg. 尖角或者梯度渐变,也被叫做过滤器)

# 第二个概念是池化层。池化层会在我们的数据上移动一个窗口(通常窗口是按每n个像素作为一个步长来移动的,叫作striding),然后对窗口里的数据以某种
# 方式求和,以缩减数据规模。最常用的方法是最大池化(max pooling),它把每个窗口的最大值传递到下一层。使用最大池化的原因之一是它很实用,卷积过程产生
# 了很多要学习的参数,这会让学习过程很快就没有什么收获,所以通过最大池化减少参数的数量是有益的。可以直观地将最大池化想象成"缩小"图像。
# 举例: 假设我们有一张包括一只狗的脸的图像。第一个卷积层可能找到一些模式,比如形状的边缘。接着,我们使用最大池化层来"缩小"图像,然后用第二个卷积层
# 找到另一些模式,比如狗的耳朵。最后,我们使用第三个最大池化层进一步缩小图像,在使用一个卷积层来找到像狗的脸这样的模式。 全连接层经常被用在网络的最后
# 来做分类。

# 解决方案中,MNIST数据集,实际上是机器学习领域的一个指标性数据集。MNIST数据集包括70,000张手写的0~9的数字的小图像(28x28像素)。数据集已经做过标记,
# 所以我们知道每一张小图像上的真实数字。标准的训练集和测试集数据配比是 用60,000张图像来做训练,用10,000张图像做测试。
# 我们把数据重新组织成卷积神经网络期望的格式,具体来讲,就是使用reshape把样本数据转换成Keras期望的形状。接着,我们把数据的值调成0和1之间,因为如果
# 样本值比网络的参数(通常都初始化成较小的数值)大很多,训练后网络的性能会很差。最后我们把目标数据用one-hot编码,这样样本的目标就有10个分类,代表0到9的数字
# 像这样处理图像数据之后,我们就可以创建卷积神经网络了。第一步,添加一个卷积层,并指定过滤器的数量和其他特性。窗口的大小是一个超参数,不过3x3的窗口
# 对于大部分的图像都适用。一般图像越大,使用的窗口就越大。 第二步,添加最大池化层,对相邻的像素求和。第三步,添加Dropout层来减小过拟合的概率。第四步,
# 添加一个压平层把卷积输入转换成全连接可用的格式。最后添加全连接层和输出层对数据进行分类。因为是多元分类问题,所以使用softmax函数。

# 20.16 通过图像增强来改善卷积神经网络的性能
from keras.preprocessing.image import ImageDataGenerator

# 创建图像增强对象
augmentation = ImageDataGenerator(featurewise_center=True,      # 实施ZCA白化
                                  zoom_range=0.3,   # 随机放大图像
                                  width_shift_range=0.2,    # 随机打乱图像
                                  horizontal_flip=True,     # 随机翻转图像
                                  rotation_range=90)        # 随机旋转图像

augment_images = augmentation.flow_from_directory("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/raw/images",      # 图像文件夹
                                                  batch_size=32,    # 批次的大小
                                                  class_mode="binary",      # 分类
                                                  save_to_dir="/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/processed/images")

# 改善卷积神经网络性能的一个方法就是预处理图像。Keras 的 ImageDataGenerator 包括一些基础的预处理技术。比如,我们的解决方案使用了featurewise_center=True
# 来标准化整个数据集中的像素。第二个改善卷积神经网络性能的技术是加入噪声。神经网络有一个有趣的特征,就是在数据中加入噪声后,网络的性能反而会变好。这是因为
# 添加的噪声可以让是神经网络对真实世界的噪声变得更具鲁棒性,并且能防止神经网络过拟合。
# 当对图像使用卷积神经网络进行训练的时候,可以通过多种方法随机转换图像,以实现向样本数据加入噪声,
# 比如镜像反转图像,或者局部方法图像。即使很小的变化,也可以改善模型的性能。flow_from_directory的输出是一个Python的生成器对象。这是因为
# 大多数情况下我们希望在需要时,即当图像被输入神经网络中进行训练的时候,才进行处理。如果想在训练之前先处理所有图像,可以简单地遍历生成器。

# 因为augment_images 是一个生成器,所以在训练神经网络时,必须使用fit_generator 而不是fit
# 举例-
# network.fit_generator(augment_images,
#                       steps_per_epoch=2000,     # 在每个epoch中调用生成器的次数
#                       epochs=5,                 # epochs 数量
#                       validation_data=augment_images_test,      # 测试数据生成器
#                       validation_steps=800)      # 在每个测试epoch中调用生成器的次数


# 20.17 文本分类
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models
from keras import layers
np.random.seed(0)
number_of_features = 1000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

features_train = sequence.pad_sequences(data_train, maxlen=400)     # 采用添加填充充值或者截断的方式,使每个样本都有400个特征
features_test = sequence.pad_sequences(data_test, maxlen=400)

network = models.Sequential()
network.add(layers.Embedding(input_dim=number_of_features, output_dim=128))     # 添加嵌入层
network.add(layers.LSTM(units=128))     # 添加一个有128个神经元的长短期记忆网络层
network.add(layers.Dense(units=1, activation="sigmoid"))        # 添加使用sigmoid激活函数的全连接层

network.compile(loss="binary_crossentropy",
                optimizer="Adam",
                metrics=["accuracy"])

history = network.fit(features_train,
                      target_train,
                      epochs=3,
                      verbose=0,
                      batch_size=1000,
                      validation_data=(features_test,target_test))      # 测试数据

# 递归神经网络。有一个关键特性:即信息在网络中回环,折让递归神经网络拥有一种记忆,使它可以更好地理解顺序数据。一种流行的递归神经是长短期记忆递归
# (Long Short-Term Memory,LSTM)神经网络。

print(data_train[0])
# 列表中每个整数对应一个词,但是因为每条影评包含的词的数量不一样,所以对应样本长度不一。因此,在将这些数据输入神经网络之前,需要让所有样本具有相同
# 的长度。我们使用pad_sequences来实现这一点。pad_sequences 为每个样本加入填充值使它们的长度相同。可以查看第一个样本经过pad_sequences 处理之后的效果
print(features_test[0])
# 接着我们使用自然语言处理领域效果最好的技术之一:词向量。这里我们把每个词表示为一个多维空间的向量,并且用两个向量之间的距离表示其对应的两个词的相似度。
# Keras中,可以通过加入一个Embedding 层实现。对于每一个传入Embedding层的值,Keras都会输出一个向量来表示这个词。紧接着的一层有128个神经元的LSTM层,
# 它使之前输入的信息能在未来适用,非常适合处理顺序数据。最后因为这个是二元分类问题,所以加入一个神经元和sigmoid激活函数全连接输出层。

