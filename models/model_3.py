
NUM_EPOCHS = 20
NUM_FOLDS = 5


import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class BobRossSegmentedImagesDataset(Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.dataroot = dataroot
        self.imgs = list((self.dataroot / 'train' / 'images').rglob('*.png'))
        self.segs = list((self.dataroot / 'train' / 'labels').rglob('*.png'))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), transforms.ToTensor()
        ])
        self.color_key = {
            3 : 0,
            5: 1,
            10: 2,
            14: 3,
            17: 4,
            18: 5,
            22: 6,
            27: 7,
            61: 8
        }
        assert len(self.imgs) == len(self.segs)
        # TODO: remean images to N(0, 1)?
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        def translate(x):
            return self.color_key[x]
        translate = np.vectorize(translate)
        
        img = Image.open(self.imgs[i])
        img = self.transform(img)
        
        seg = Image.open(self.segs[i])
        seg = seg.resize((256, 256))
        
        # Labels are in the ADE20K ontology and are not consequetive,
        # we have to apply a remap operation over the labels in a just-in-time
        # manner. This slows things down, but it's fine, this is just a demo
        # anyway.
        seg = translate(np.array(seg)).astype('int64')
        
        # One-hot encode the segmentation mask.
        # def ohe_mat(segmap):
        #     return np.array(
        #         list(
        #             np.array(segmap) == i for i in range(9)
        #         )
        #     ).astype(int).reshape(9, 256, 256)
        # seg = ohe_mat(seg)
        
        # Additionally, the original UNet implementation outputs a segmentation map
        # for a subset of the overall image, not the image as a whole! With this input
        # size the segmentation map targeted is a (164, 164) center crop.
        seg = seg[46:210, 46:210]
        
        return img, seg
    
    
from torch import nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1_1 = nn.Conv2d(3, 64, 3)
        self.relu_1_2 = nn.ReLU()
        self.conv_1_3 = nn.Conv2d(64, 64, 3)
        self.relu_1_4 = nn.ReLU()
        self.pool_1_5 = nn.MaxPool2d(2)
        
        self.conv_2_1 = nn.Conv2d(64, 128, 3)
        self.relu_2_2 = nn.ReLU()
        self.conv_2_3 = nn.Conv2d(128, 128, 3)
        self.relu_2_4 = nn.ReLU()        
        self.pool_2_5 = nn.MaxPool2d(2)
        
        self.conv_3_1 = nn.Conv2d(128, 256, 3)
        self.relu_3_2 = nn.ReLU()
        self.conv_3_3 = nn.Conv2d(256, 256, 3)
        self.relu_3_4 = nn.ReLU()
        self.pool_3_5 = nn.MaxPool2d(2)
        
        self.conv_4_1 = nn.Conv2d(256, 512, 3)
        self.relu_4_2 = nn.ReLU()
        self.conv_4_3 = nn.Conv2d(512, 512, 3)
        self.relu_4_4 = nn.ReLU()
        
        # deconv is the '2D transposed convolution operator'
        self.deconv_5_1 = nn.ConvTranspose2d(512, 256, (2, 2), 2)
        # 61x61 -> 48x48 crop
        self.c_crop_5_2 = lambda x: x[:, :, 6:54, 6:54]
        self.concat_5_3 = lambda x, y: torch.cat((x, y), dim=1)
        self.conv_5_4 = nn.Conv2d(512, 256, 3)
        self.relu_5_5 = nn.ReLU()
        self.conv_5_6 = nn.Conv2d(256, 256, 3)
        self.relu_5_7 = nn.ReLU()
        
        self.deconv_6_1 = nn.ConvTranspose2d(256, 128, (2, 2), 2)
        # 121x121 -> 88x88 crop
        self.c_crop_6_2 = lambda x: x[:, :, 17:105, 17:105]
        self.concat_6_3 = lambda x, y: torch.cat((x, y), dim=1)
        self.conv_6_4 = nn.Conv2d(256, 128, 3)
        self.relu_6_5 = nn.ReLU()
        self.conv_6_6 = nn.Conv2d(128, 128, 3)
        self.relu_6_7 = nn.ReLU()
        
        self.deconv_7_1 = nn.ConvTranspose2d(128, 64, (2, 2), 2)
        # 252x252 -> 168x168 crop
        self.c_crop_7_2 = lambda x: x[:, :, 44:212, 44:212]
        self.concat_7_3 = lambda x, y: torch.cat((x, y), dim=1)
        self.conv_7_4 = nn.Conv2d(128, 64, 3)
        self.relu_7_5 = nn.ReLU()
        self.conv_7_6 = nn.Conv2d(64, 64, 3)
        self.relu_7_7 = nn.ReLU()
        
        # 1x1 conv ~= fc; n_classes = 9
        self.conv_8_1 = nn.Conv2d(64, 9, 1)

    def forward(self, x):
        x = self.conv_1_1(x)
        x = self.relu_1_2(x)
        x = self.conv_1_3(x)
        x_residual_1 = self.relu_1_4(x)
        x = self.pool_1_5(x_residual_1)
        
        x = self.conv_2_1(x)
        x = self.relu_2_2(x)        
        x = self.conv_2_3(x)
        x_residual_2 = self.relu_2_4(x)        
        x = self.pool_2_5(x_residual_2)
        
        x = self.conv_3_1(x)
        x = self.relu_3_2(x)        
        x = self.conv_3_3(x)
        x_residual_3 = self.relu_3_4(x)
        x = self.pool_3_5(x_residual_3)
        
        x = self.conv_4_1(x)
        x = self.relu_4_2(x)
        x = self.conv_4_3(x)
        x = self.relu_4_4(x)
        
        x = self.deconv_5_1(x)
        x = self.concat_5_3(self.c_crop_5_2(x_residual_3), x)
        x = self.conv_5_4(x)
        x = self.relu_5_5(x)
        x = self.conv_5_6(x)
        x = self.relu_5_7(x)
        
        x = self.deconv_6_1(x)
        x = self.concat_6_3(self.c_crop_6_2(x_residual_2), x)
        x = self.conv_6_4(x)
        x = self.relu_6_5(x)
        x = self.conv_6_6(x)
        x = self.relu_6_7(x)
        
        x = self.deconv_7_1(x)
        x = self.concat_7_3(self.c_crop_7_2(x_residual_1), x)
        x = self.conv_7_4(x)
        x = self.relu_7_5(x)
        x = self.conv_7_6(x)
        x = self.relu_7_7(x)
        
        x = self.conv_8_1(x)
        return x

from pathlib import Path
dataroot = Path('/spell/bob-ross-kaggle-dataset/')
dataset = BobRossSegmentedImagesDataset(dataroot)
dataloader = DataLoader(dataset, shuffle=True)

import numpy as np
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

kf = KFold(n_splits=NUM_FOLDS, shuffle=True)
idxs = list(range(len(dataset)))
for fold, (train_idxs, test_idxs) in enumerate(kf.split(idxs)):
    writer = SummaryWriter(f'/spell/tensorboards/experiment_3_fold_{fold}')
    model = UNet()
    model.cuda()
    
    for epoch in range(NUM_EPOCHS):
        losses = []

        for i, train_idx in enumerate(train_idxs):
            batch, segmap = dataset[i]

            batch = batch[None].cuda()
            segmap = torch.tensor(segmap[None]).cuda()

            optimizer.zero_grad()

            output = model(batch)
            loss = criterion(output, segmap)
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            if i % 50 == 0:
                print(
                    f'Finished fold {fold}, epoch {epoch}, batch {i}. Loss: {curr_loss:.3f}.'
                )

            writer.add_scalar(
                'training loss', curr_loss, epoch * len(dataloader) + i
            )
            losses.append(curr_loss)

        print(
            f'Finished epoch {epoch}. '
            f'avg loss: {np.mean(losses)}; median loss: {np.min(losses)}'
        )
