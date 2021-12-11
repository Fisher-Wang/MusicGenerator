from pickle import NONE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

music_length = 32

class CYQNN(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.fc1 = nn.Linear(music_length, 2 * music_length, bias=False)
        self.sig1 = torch.nn.Sigmoid()
        self.fc2 = nn.Linear(2 * music_length, 2 * music_length, bias=False)
        self.sig2 = torch.nn.Sigmoid()
        self.fc3 = nn.Linear(2 * music_length, 2, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.fc1.weight.data.normal_(0, 1)
        self.fc2.weight.data.normal_(0, 1)
        self.fc3.weight.data.normal_(0, 1)
        # self.fc4.weight.data.normal_(0, 0.02)
        # self.fc5.weight.data.normal_(0, 0.02)
        # print(self.fc1.weight.data)
        # print(self.fc2.weight.data)
        # print(self.fc3.weight.data)

    def forward(self, x):
        out = self.sig1(self.fc1(x))
        out = self.sig2(self.fc2(out))
        # out = self.sig3(self.fc3(out))
        # out = self.sig4(self.fc4(out))
        out = self.fc3(out)
        out = self.softmax(out)
        # out = self.relu2(self.fc2(out))
        # out = self.relu3(self.fc3(out))
        # out = self.fc4(out)
        # out = self.fc2(out)
        # out = self.sig(out)
        return out



from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

normalize = transforms.Normalize(
    mean=[0.5],
    std=[0.23]
)
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #normalize
])

def default_loader(music):
    #music_tensor = preprocess(music)
    music = torch.tensor(music)
    music = music.to(torch.float32).reshape(music_length)
    music = music / 100 - 0.6
    return music

#当然出来的时候已经全都变成了tensor
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        self.music = np.load("./dataset/net/notes.npy")
        self.target = torch.Tensor(np.load("./dataset/net/targets.npy")).long()
        self.loader = loader
        
        

    def __getitem__(self, index):
        music_seg = self.music[index]
        music_tensor = self.loader(music_seg)
        target = self.target[index]
        return music_tensor,target

    def __len__(self):
        return len(self.music)




net=CYQNN()



optimizer = torch.optim.Adagrad(net.parameters(),lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

train_data  = trainset()

trainloader = DataLoader(train_data, batch_size=1,shuffle=True)


# pre_data1 = copy.deepcopy(net.fc1.weight.data)
# pre_data2 = copy.deepcopy(net.fc2.weight.data)


for epoch in range(500):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the input
        inputs, labels = data

        
        # zeros the paramster gradients
        optimizer.zero_grad()       # 

        # forward + backward + optimize
        outputs = net(inputs)
        #labels = labels.unsqueeze(1)
        #labels = labels.float()

        # print(inputs.size())
        # print(labels.size())

        loss = criterion(outputs, labels)  # 计算loss
        # if epoch >= 0:
        #     print(inputs)
        #     x = net.sig1(net.fc1(inputs))
        #     print(x)
        #     x = net.sig2(net.fc2(x))
        #     print(x)
        #     x = net.fc3(x)
        #     print(x)
        #     print(outputs)
        #     print(labels)
        #     print(loss)
        #     input()


            # print(net.fc1.weight.data)
            # print(net.fc2.weight.data)
            # print(net.fc3.weight.data)
            # input()
            # print(net.fc1.weight.data - pre_data1)
            # print(net.fc2.weight.data - pre_data2)
            # input()
            # pre_data1 = copy.deepcopy(net.fc1.weight.data)
            # pre_data2 = copy.deepcopy(net.fc2.weight.data)

        loss.backward()     # loss 求导
        optimizer.step()    # 更新参数

        # print statistics
        running_loss += loss.item()  # tensor.item()  获取tensor的数值
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 99))  # 每2000次迭代，输出loss的平均值
            running_loss = 0.0
            torch.save(net.state_dict(), "net{:}.pkl".format((i + 1) // 2000))

torch.save(net.state_dict(), "net_final.pkl")
        