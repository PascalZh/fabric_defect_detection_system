# -*- coding: utf-8 -*-
from PIL import Image

# import matplotlib.pyplot as plt
import cv2
import time

import torch
from torch import nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import albumentations as A

from os import system

# torch.set_num_threads(4)


class NetClassify(nn.Module):
    def __init__(self, model_name, class_number):
        super(NetClassify, self).__init__()

        self.name = model_name

        # init model
        if "eff" in model_name:
            #model = EfficientNet.from_name(model_name)
            self.model_feature = EfficientNet.from_name(
                model_name.replace('adv-', ''))

        elif "wsl" in model_name:
            model = torch.hub.load('facebookresearch/WSL-Images', model_name)

        # elif "RegNet" in model_name:

        #     model_cate = model_name.split('-')[-1]
        #     self.model_feature = pymodel.regnety(model_cate, pretrained=False)#, cfg_list=("MODEL.NUM_CLASSES", 4))

        # elif 'EN-B' in model_name:
        #     model_cate = model_name.split('-')[-1]
        #     self.model_feature = pymodel.effnet(model_cate, pretrained=False)
        #     #b

        else:
            # model_name = 'resnext50' # se_resnext50_32x4d xception
            model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained=None)
            print(pretrainedmodels.pretrained_settings[model_name])

        if "eff" in model_name:

            #self.model_feature._dropout = nn.Dropout(0.6)

            fc_features = self.model_feature._fc.in_features
            self.model_feature._fc = nn.Sequential(
                nn.Linear(fc_features, class_number))
            #self.model_feature._fc = nn.Linear(fc_features, class_number)
            #self.model_feature = nn.Sequential(*list(self.model_feature.children())[:-3])
            # print(self.svm)
            # b
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(512, 128),
            # nn.ReLU(),
            # nn.Dropout(0.50),
            # nn.Linear(128,class_number))
            # print(list(self.model_feature.children())[:-3])
            # b
            # self.model_features = nn.Sequential(*list(self.model_feature.children())[:-3])
            # self.last_linear = nn.Linear(fc_features, class_number)

        elif "RegNet" in model_name:
            # print(model)
            # print(nn.Sequential(*list(model.children())[:-1]))
            # b

            fc_features = self.model_feature.head.fc.in_features
            self.model_feature.head.fc = nn.Sequential(
                nn.Linear(fc_features, class_number))

            self.featuremap1 = x.detach()
            # for k,v in self.model_feature.named_parameters():
            #     print('{}: {}'.format(k, v.requires_grad))
            # b

        elif "EN-B" in model_name:
            # print(model)
            # print(nn.Sequential(*list(model.children())[:-1]))
            # b
            fc_features = self.model_feature.head.fc.in_features
            self.model_feature.head.fc = nn.Sequential(
                nn.Linear(fc_features, class_number))

        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model_feature = nn.Sequential(*list(model.children())[:-1])

            # self.dp_linear = nn.Linear(fc_features, 8)
            # self.dp = nn.Dropout(0.50)
            self.last_linear = nn.Linear(fc_features, class_number)
            #self.last_linear = BinaryHead(3, emb_size=2048, s=1)

        # print(self.model_feature)
        # b

    def forward(self, img):
        #self.svm = self.svm_feature(img)
        out = self.model_feature(img)
        # out = self.last_linear(out)
        #out = self.avgpool(out)
        if "RegNet" in self.name or "EN-B" in self.name:
            return out

        if self.name == "xception":
            out = self.avgpool(out)

        if "eff" not in self.name:
            out = out.view(out.size(0), -1)

            out = self.last_linear(out)

        return out


class TestDataAug:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        min_size = max(img.shape[:2])

        img = A.PadIfNeeded(min_height=min_size, min_width=min_size,
                            border_mode=3, value=0, mask_value=0,
                            always_apply=False, p=1)(image=img)['image']

        img = A.Resize(self.h, self.w, p=1)(image=img)['image']
        img = Image.fromarray(img)
        return img


category_name = {0: "normal", 1: "defection"}

model = NetClassify('adv-efficientnet-b0', 2)

transform = transforms.Compose([
    TestDataAug(224, 224),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2.0 - 1.0)
])

model.load_state_dict(torch.load(
    "./model.pth", map_location=torch.device('cpu')))
model.eval()


def predict(img_path):
    with torch.no_grad():

        print('Evaluating model...')
        img = cv2.imread(img_path)
        img_t = transform(img)
        # plt.imshow(img_t.numpy()[0])
        batch_t = torch.unsqueeze(img_t, 0)
        # print(batch_t)

        out = model(batch_t)[0]
        out = nn.functional.softmax(out, dim=0)

        print(f'Predict scores: {out}')
        idx_result = torch.argmax(out).item()
        print(f'Predict result: {category_name[idx_result]}')
        print(f'Result score: {out[idx_result]}')

    return category_name[idx_result], out[idx_result]


if __name__ == "__main__":
    predict('./tmp/captured.jpg')
# plt.show()
