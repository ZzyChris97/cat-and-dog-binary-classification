# 猫狗二分类训练模型

训练二分类模型，熟悉数据读取机制，并且从kaggle中下载猫狗二分类训练数据，编写一个DogCatDataset，使得pytorch可以对猫狗二分类训练集进行读取

## data

  data目录存放训练所用到的图片数据
  数据：链接：https://pan.baidu.com/s/1vOvxTD8LYRIDq4IGr-WZ3A  提取码：5mec

## split_dataset.py
  将数据集分为训练集，验证集和测试集
  
## mydataset.py
  实现RMBDataset中的__getitem__函数，实现根据索引返回和图片和标签，使得pytorch可以对猫狗二分类训练集进行读取。
  
## lenet.py
  模型
  
## train_lenet.py
  训练模型
