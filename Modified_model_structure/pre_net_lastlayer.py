import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Resnet_Pre_Net(nn.Module):
    def __init__(self, model):
        super(Resnet_Pre_Net, self).__init__()
        self.num_infeature = list(model.children())[-1].in_features
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])#resnet  &&  densenet
        self.fc = nn.Linear(self.num_infeature,256)
        self.classifier = nn.Sequential(
                            # nn.Linear(input_num, 256),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(256, 2),
                            nn.LogSoftmax(dim=1)
                           )

    def forward(self, x):

        x = self.resnet_layer(x)

        x = self.fc(x)

        x = self.classifier(x)

        return x

class Densenet_Pre_Net(nn.Module):
    def __init__(self, model):
        super(Densenet_Pre_Net, self).__init__()
        # 取掉model的后两层
        # self.num_infeature = list(model.children())[-1].in_features
        self.num_infeature = 512
        # self.resnet_layer = nn.Sequential(*list(model.children())[:-1])#resnet  &&  densenet
        self.resnet_layer = nn.Sequential(*list(model.children()))[0][:6]
        self.Avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.num_infeature,256)
        self.classifier = nn.Sequential(
                            # nn.Linear(input_num, 256),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(256, 2),
                            nn.LogSoftmax(dim=1)
                           )

    def forward(self, x):

        x = self.resnet_layer(x)
        x = self.Avgpool(x)
        x = torch.squeeze(x,dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc(x)

        x = self.classifier(x)

        return x

class VGG_Pre_Net(nn.Module):
    def __init__(self, model):
        super(VGG_Pre_Net, self).__init__()
        # 取掉model的后两层
        # self.num_infeature = list(model.children())[-1].fc.in_features  # vgg16
        self.num_infeature = 512
        # self.resnet_layer = nn.Sequential(*list(model.children())[:-1])#resnet  &&  densenet
        self.resnet_layer = nn.Sequential(*list(model.children())[0][:23])  # resnet  &&  densenet
        self.Avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.num_infeature,256)
        self.classifier = nn.Sequential(
                            # nn.Linear(input_num, 256),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(256, 2),
                            nn.LogSoftmax(dim=1)
                           )

    def forward(self, x):

        x = self.resnet_layer(x)
        x = self.Avgpool(x)
        x = torch.squeeze(x, dim=2)#vgg
        x = torch.squeeze(x, dim=2)#vgg
        x = self.fc(x)
        x = self.classifier(x)

        return x

class Mobile_Pre_Net(nn.Module):
    def __init__(self, model):
        super(Mobile_Pre_Net, self).__init__()
        # 取掉model的后两层
        self.num_infeature = list(model.children())[-1].in_features
        self.resnet_layer = nn.Sequential(*list(model.children())[:-5])
        # self.resnet_layer = nn.Sequential(*list(model.children())[:-1])#resnet  &&  densenet
        # self.num_infeature = list(model.children())[-1].fc.in_features#vgg16
        self.mobile_second = nn.Sequential(*list(model.children())[-5:-1])
        self.fc = nn.Linear(self.num_infeature,256)
        self.classifier = nn.Sequential(
                            # nn.Linear(input_num, 256),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(256, 2),
                            nn.LogSoftmax(dim=1)
                           )


    def forward(self, x):

        x = self.resnet_layer(x)
        x = self.mobile_second(x)
        x = self.fc(x)

        x = self.classifier(x)

        return x

class Conv_Pre_Net(nn.Module):
    def __init__(self, model):
        super(Conv_Pre_Net, self).__init__()
        # 取掉model的后两层
        self.num_infeature = list(model.children())[-1].fc.in_features
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        # self.resnet_layer = nn.Sequential(*list(model.children())[:-1])#resnet  &&  densenet
        # self.num_infeature = list(model.children())[-1].fc.in_features#vgg16
        self.mobile_second = nn.Sequential(*list(model.children())[-1][:-1])
        self.fc = nn.Linear(self.num_infeature,256)
        self.classifier = nn.Sequential(
                            # nn.Linear(input_num, 256),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(256, 2),
                            nn.LogSoftmax(dim=1)
                           )


    def forward(self, x):

        x = self.resnet_layer(x)
        x = self.mobile_second(x)
        x = self.fc(x)

        x = self.classifier(x)

        return x


class Xception_Pre_Net(nn.Module):
    def __init__(self, model):
        super(Xception_Pre_Net, self).__init__()
        # 取掉model的后两层
        # self.num_infeature = list(model.children())[-1].in_features
        self.num_infeature = 288
        self.resnet_layer = nn.Sequential(*list(model.children())[:10])
        # self.resnet_layer = nn.Sequential(*list(model.children())[:-1])#resnet  &&  densenet
        # self.num_infeature = list(model.children())[-1].fc.in_features#vgg16
        # self.mobile_second = nn.Sequential(*list(model.children())[-1][:-1])
        self.Avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.num_infeature,256)
        self.classifier = nn.Sequential(
                            # nn.Linear(input_num, 256),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(256, 2),
                            nn.LogSoftmax(dim=1)
                           )


    def forward(self, x):

        x = self.resnet_layer(x)
        x = self.Avgpool(x)
        x = torch.squeeze(x,dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc(x)

        x = self.classifier(x)

        return x