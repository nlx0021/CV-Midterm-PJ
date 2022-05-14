# CV-Midterm-PJ
这是我们的计算机视觉期中项目代码。

## 小组成员
李文烨（19307110305）
秦子述（18307110252）
胡诚驿（19300180073）

## Part1：CIFAR-100
第一部分的代码在此Repo的“Part1:CIFAR-100”Branch分支上。
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

## Part2：Object Detection
第二部分的代码在此Repo的“Part2:Objects Detection”Branch分支上。
### 模型训练
