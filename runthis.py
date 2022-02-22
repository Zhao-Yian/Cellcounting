# -*-coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import os
import glob
# 读取图像
#获取dataset2-master目录下的所有子目录名称，并分类保存路径
test_img_list = glob.glob("/home/aistudio/work/dataset2-master/dataset2-master/images/TEST/*/*.jpeg")
test_simple_img_list = glob.glob("/home/aistudio/work/dataset2-master/dataset2-master/images/TEST_SIMPLE/*/*.jpeg")
train_img_list = glob.glob("/home/aistudio/work/dataset2-master/dataset2-master/images/TRAIN/*/*.jpeg")
print(len(test_img_list), len(test_simple_img_list), len(train_img_list))
#获取dataset-master目录下的所有子目录名称，保存进一个列表之中(暂时不做处理，未来将会将其中图片用于目标检测任务)
img_path_list = os.listdir("/home/aistudio/work/dataset-master/dataset-master/JPEGImages")
img_list = []
for img_path in img_path_list:
    img_list.append("/home/aistudio/work/dataset-master/dataset-master/JPEGImages/"+img_path)
# 可视化部分图像
for i in range(6):
    plt.subplot(2,3,i+1)
    img_bgr = cv2.imread(img_list[i])
    img_gbr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_gbr)

# 计算图像数据整体的均值和方差
import tqdm
import glob
import numpy as np

def get_mean_std(image_path_list):
    print('Total images size:', len(image_path_list))
    # 结果向量的初始化
    max_val, min_val = np.zeros(3), np.ones(3) * 255
    mean, std = np.zeros(3), np.zeros(3)

    for image_path in tqdm.tqdm(image_path_list):
        image = cv2.imread(image_path)
        for c in range(3):
            # 计算每个通道的均值和方差
            mean[c] += image[:, :, c].mean()
            std[c] += image[:, :, c].std()
            max_val[c] = max(max_val[c], image[:, :, c].max())
            min_val[c] = min(min_val[c], image[:, :, c].min())

    # 图像的平均均值和方差
    mean /= len(image_path_list)
    std /= len(image_path_list)

    mean /= max_val - min_val
    std /= max_val - min_val
    # print(max_val - min_val)
    return mean, std

image_path_list = []
# image_path_list.extend(glob.glob('/home/aistudio/work/dataset-master/dataset-master/JPEGImages/*.jpg'))
image_path_list.extend(glob.glob('/home/aistudio/work/dataset2-master/dataset2-master/images/*/*/*.jpeg'))
mean, std = get_mean_std(image_path_list)
print('mean:', mean)
print('std:', std)

#数据预处理(提高色彩对比度，便于神经网络提取特征)
import cv2
import paddle
import numpy as np
import paddle.vision.transforms as T

def preprocess(img):
    transform = T.Compose([
        T.Resize(size=(240, 240)),
        T.ToTensor(),
        T.Normalize(mean=[0.66109358, 0.64167306, 0.67932446], std=[0.25564928, 0.25832876, 0.25864613])
        ])
    img = transform(img).astype('float32')
    return img

img_bgr = cv2.imread('work/dataset2-master/dataset2-master/images/TRAIN/EOSINOPHIL/_0_6149.jpeg')
img = preprocess(img_bgr)
print(img.shape, img_bgr.shape)
plt.imshow(img.T)

# 数据集
class MyImageDataset(paddle.io.Dataset):
    def __init__(self, phase = 'train'):
        super(MyImageDataset, self).__init__()
        assert phase in ['train', 'test']
        self.samples = []
        if (phase == 'train'):
            self.samples.extend(glob.glob('/home/aistudio/work/dataset2-master/dataset2-master/images/TRAIN/*/*.jpeg'))
        else:
            self.samples.extend(glob.glob('/home/aistudio/work/dataset2-master/dataset2-master/images/TEST/*/*.jpeg'))
        self.labels = {'EOSINOPHIL': 0, 'LYMPHOCYTE': 1, 'MONOCYTE': 2, 'NEUTROPHIL': 3}

    def __getitem__(self, index):
        img_path = self.samples[index]
        img_bgr = cv2.imread(img_path)
        image = preprocess(img_bgr)
        label = self.labels[img_path.split('/')[-2]]
        label = paddle.to_tensor([label], dtype='int64')
        return image, label

    def __len__(self):
        return len(self.samples)

train_dataset = MyImageDataset('train')
test_dataset = MyImageDataset('test')
print(len(train_dataset), len(test_dataset))
print(train_dataset[0][0].shape, test_dataset[0][0].shape)

# 定义超参数
lr = 0.0005
BATCH_SIZE = 64
EPOCH = 10

#定义数据读取器
train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle= True)
for batch_id, data in enumerate(train_loader()):
    x_data = data[0]
    y_data = data[1]
    print(batch_id)
    print(x_data.shape)
    print(y_data.shape)
    break

#图像格式预处理(data_fromat:'NCHW')
# 定义网络
import paddle.nn as nn
resnet = paddle.vision.models.resnet18()
resnet = nn.Sequential(*list(resnet.children())[:-1],
            nn.Flatten(),
            nn.Linear(512,10))
paddle.summary(resnet,(64, 3, 240, 240))
#定义优化器和损失函数
opt = paddle.optimizer.Adam(
    learning_rate=lr,
    parameters=resnet.parameters(),
    beta1=0.5,
    beta2=0.999
)

loss_func = paddle.nn.CrossEntropyLoss()

# 开始训练
from tqdm import tqdm
resnet.train()
for epoch in range(EPOCH):
    for batch_id, data in tqdm(enumerate(train_loader())):
        imgs = data[0]
        label = data[1]
        infer_prob = resnet(imgs)
        loss = loss_func(infer_prob, label)
        acc = paddle.metric.accuracy(infer_prob, label)
        loss.backward()
        opt.step()
        opt.clear_gradients()
        if (batch_id + 1) % 20 == 0:
            print('第{}个epoch:batch_id is: {},loss is: {}, acc is: {}'.format(epoch,batch_id + 1, loss.numpy(),acc.numpy()))
paddle.save(resnet.state_dict(), 'param')

# 测试集
resnet.set_state_dict(paddle.load("param"))
resnet.eval()
global_acc = 0
idx = 0
for batch_id, testdata in enumerate(test_loader()):
    imgs = testdata[0]
    label = testdata[1]
    predicts = resnet(imgs)
    # 计算损失与精度
    loss = loss_func(predicts, label)
    global_acc += paddle.metric.accuracy(predicts, label).numpy()[0]
    idx += 1
    # 打印信息
    if (batch_id+1) % 20 == 0:
        print("batch_id: {}, loss is: {}".format(batch_id+1, loss.numpy()))
print("全局准确率为：{}".format(global_acc / idx))