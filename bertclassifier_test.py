
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from BERTClassifier import MyBERTClassier
from sklearn.model_selection import train_test_split
import pandas as pd


# https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset

# 为了节省时间我就只采样5000条
file1 = 'C:/bd_ai/dli/models/bert/toutiao_cat_data.txt'
#file1 = 'C:/bd_ai/dli/models/bert/123.txt'
corpusDf = pd.read_csv(file1, sep='_!_', encoding='utf8', engine='python', header=0).sample(50)
#corpusDf.drop(["newsId", "newsNum", "loc"], axis=1, inplace=True)
#corpusDf = corpusDf.loc[:, ["text", "class"]]
corpusDf = corpusDf.loc[:, ['content1', 'clsname']]
corpusDf.head()

# text列应以list形式呈现并在开头与结尾加上"[CLS]"和"[SEP]"，并记录下最长句子
corpusDf["content1"] = corpusDf["content1"].apply(lambda row: ["[CLS]"]+list(row)+["[SEP]"])
maxLen = max([len(row) for row in corpusDf["content1"]])
print("最大长度：{}".format(maxLen))

corpusDf.reset_index(drop=True)["content1"][0]

# 类别应变为[0,1,2,...]的格式，同时记录下类别的数目
classTuple = tuple(set(corpusDf["clsname"].values.tolist()))
classNum = len(classTuple)
print("分类数目：{}".format(classNum))
print(classTuple)

corpusDf["clsname"] = corpusDf["clsname"].apply(lambda row: classTuple.index(row))

corpus = corpusDf.values

print(corpusDf.head())
# print(corpus)

# 区分训练集和测试集
trainX, testX, trainY, testY = train_test_split(corpus[:, 0], corpus[:, 1], test_size=0.3)

# 构造模型
myModel = MyBERTClassier(classNum=classNum, XMaxLen=maxLen, learning_rate=0.0001)

# 训练模型
myModel.fit(trainX, trainY, epochs=15, batch_size=16)

# 预测数据
result = np.argmax(myModel.predict(testX), axis=-1)
print(result)
