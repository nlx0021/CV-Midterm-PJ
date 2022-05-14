# CV-Midterm-PJ
这是我们的计算机视觉期中项目代码。

## 小组成员
李文烨（19307110305）
秦子述（18307110252）
胡诚驿（19300180073）

## Part1：CIFAR-100
第一部分的代码在此Repo的“Part1:CIFAR-100”Branch分支上。
### 环境部署
在进行运行前，请先将external.zip压缩文件解压到当前文件夹当中，解压得到external文件夹。
### 模型训练
模型的训练在train.py文件中。训练模型可运行以下命令：
```
python train.py --experiment experiment_name --lr 5e-3 --epochs_n 100
```
训练时会在tensorboard文件夹中生成一个名字为experiment_name的文件夹，里面记录了tensorboard的event文件，可运行此命令查看：
```
tensorboard --logdir tensorboard/experiment_name
```
另外，训练结束后，与这次实验相关的信息会记录在experiment文件夹中的experiment_name文件夹中，里面包含相关的曲线以及训练信息（文本文件）。
  
训练结束后，模型会保存在model文件夹下。
  
可以调节训练的具体参数，包括设置是否使用cutmix、mixup以及cutout。若想进行设置，请进入train.py文件内修改，请在对应位置进行修改（有注释）。

### 模型测试
若要测试模型，请运行test.py文件。若要对model文件夹下的example.pth模型进行测试，请运行以下命令：
```
python test.py --model example.pth
```
运行后将会输出测试的结果。

## Part2：Object Detection: Faster-Rcnn
第二部分关于Faster-Rcnn的代码在此Repo的“Part2:Objects Detection: Faster-rcnn”Branch分支上。

### 数据下载
VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：
链接: https://pan.baidu.com/s/1YuBbBKxm2FGgTU5OfaeC5A
提取码: uack


预训练好的模型下载地址如下，下载后请放在相应模型的/model_data/目录下。
连接：https://pan.baidu.com/s/1-iy\_HXjzSOyDZigLApZTew?pwd=0514
提取码：0514

### 模型训练
1. 数据集的准备：下载VOC数据集，解压后放在根目录。
2. 开始网络训练：train_fix.py的默认参数用于训练VOC数据集，直接运行train_fix.py开始训练。
3. 指定参数训练：您也可以自己指定训练参数，详情请见train_fix.py文件中的注释。
4. 模型文件存储：训练好的模型权重以及训练过程的学习曲线会输出到/logs/路径下。

### 对图片进行测试
#### a. 使用预训练模型权重
1. 从百度网盘下载预训练权重，放在/model_data/目录下。
2. 运行predicy.py，输入img/street.jpg（或使用img文件夹下的其它图片）。

#### b. 使用预训练模型权重
1. 按照训练步骤训练。
2. 编辑frcnn.py文件中的model_path，使其指向/logs/路径下对应的模型权重文件。
3. 运行predict.py，输入img/street.jpg（或使用img文件夹下的其它图片）。

## Part2：Object Detection：YOLO V3
第二部分关于YOLO的代码在此Repo的“Part2:Objects Detection: YOLO”Branch分支上。

### 数据下载
VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：
链接: https://pan.baidu.com/s/1YuBbBKxm2FGgTU5OfaeC5A
提取码: uack


预训练好的模型下载地址如下，下载后请放在相应模型的/model_data/目录下。
连接：https://pan.baidu.com/s/1-iy\_HXjzSOyDZigLApZTew?pwd=0514
提取码：0514
（同上）

### 模型训练
1. 数据集的准备：下载VOC数据集，解压后放在根目录。
2. 开始网络训练：train_fix.py的默认参数用于训练VOC数据集，直接运行train_fix.py开始训练。
3. 指定参数训练：您也可以自己指定训练参数，详情请见train_fix.py文件中的注释。
4. 模型文件存储：训练好的模型权重以及训练过程的学习曲线会输出到/logs/路径下。


### 对图片进行测试
#### a. 使用预训练模型权重
1. 从百度网盘下载预训练权重，放在/model_data/目录下。
2. 运行predicy.py，输入img/street.jpg（或使用img文件夹下的其它图片）。

#### b. 使用预训练模型权重
1. 按照训练步骤训练。
2. 编辑yolo件中的model_path，使其指向/logs/路径下对应的模型权重文件。
3. 运行predict.py，输入img/street.jpg（或使用img文件夹下的其它图片）。

