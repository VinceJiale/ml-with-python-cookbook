# 21.1 保存和加载一个scikit-learn 模型
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.externals import joblib

iris = datasets.load_iris()
features = iris.data
target = iris.target
classifier = RandomForestClassifier()
model = classifier.fit(features, target)

joblib.dump(model,"model.pkl")      # 把模型保存为pickle文件

classifier = joblib.load("python 机器学习手册/model.pkl")     # 从文件中加载模型

new_observation = [[5.2, 3.2, 1.1, 0.1]]        # 创建新的样本
classifier.predict(new_observation)         # 预测样本的分

# 把模型保存成一个文件,使它可以被另一个应用或者工作流加载。把模型保存为pickle文件就能达到目的。pickle是python特有的数据格式
# 具体来讲我们使用joblib(它是一个库,作用是让pickle文件使用于Numpy数组很大的情况下,在scikit-learn 训练后的模型中,这种情况很常见)来保存模型

# 当保存scikit-learn模型时,要留心,因为你所保存的模型很有可能在各个版本的scikit-learn 中不兼容,所以赞文件名中写上模型所用的scikit-learn版本就会很有用:
import sklearn
scikit_version = sklearn.__version__     # 获取scikit-learn的版本
joblib.dump(model, "model_{version}.pkl".format(version=scikit_version))

# 21.2 保存和加载Keras 模型
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.models import load_model
np.random.seed(0)
number_of_features = 1000
(train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=number_of_features)
tokenizer = Tokenizer(num_words=number_of_features)
train_features = tokenizer.sequences_to_matrix(train_data, mode="binary")
test_features = tokenizer.sequences_to_matrix(test_data, mode="binary")
network = models.Sequential()
network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))
network.add(layers.Dense(units=1, activation="sigmoid"))
network.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
history = network.fit(train_features,
                      train_target,
                      epochs=3,
                      verbose=0,
                      batch_size=100,
                      validation_data=(test_features, test_target))

network.save("model.h5")            # 保存神经网络

network = load_model("model.h5")    # 加载神经网络
# Keras 要将模型保存为HDF5文件,包含了你需要的一切,不仅包括加载模型做预测所需要的结构和训练后的参数,而且包括重新训练所需要的各种设置
# (即损失,优化器的设置和当前状态)