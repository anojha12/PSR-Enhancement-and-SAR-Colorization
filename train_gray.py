import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#%matplotlib inline
import statistics
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

torch.cuda.device_count()

class Gray(object):
    def __call__(self, img):
        gray = img.convert('L')
        return gray

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def load_datasets():
    SAR_transform  = transforms.Compose([
    Gray(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    opt_transform  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5,), std=(0.5,0.5,0.5,))
    ])
    # s1フォルダにはSAR画像のセット、s2フォルダには光学画像のセットが入っています。
    SAR_trainsets = datasets.ImageFolder(root = '/home/anojha/pp/clr',transform=SAR_transform)
    opt_trainsets = datasets.ImageFolder(root = '/home/anojha/pp/clr',transform=opt_transform)
    Image_datasets = ConcatDataset(SAR_trainsets,opt_trainsets)
    train_loader = torch.utils.data.DataLoader(
             Image_datasets,
             batch_size=8, shuffle=True,
             num_workers=2, pin_memory=True)
    return train_loader

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_bn_relu(1, 64, kernel_size=5)
        self.enc2 = self.conv_bn_relu(64, 128, kernel_size=3, pool_kernel=4)
        self.enc3 = self.conv_bn_relu(128, 256, kernel_size=3, pool_kernel=2)
        self.enc4 = self.conv_bn_relu(256, 512, kernel_size=3, pool_kernel=2)

        self.dec1 = self.conv_bn_relu(512, 256, kernel_size=3, pool_kernel=-2,flag=True,enc=False)
        self.dec2 = self.conv_bn_relu(256+256, 128, kernel_size=3, pool_kernel=-2,flag=True,enc=False)
        self.dec3 = self.conv_bn_relu(128+128, 64, kernel_size=3, pool_kernel=-4,enc=False)
        self.dec4 = nn.Sequential(
            nn.Conv2d(64 + 64, 3, kernel_size=5, padding=2), # padding=2にしているのは、サイズを96のままにするため
            nn.Tanh()
        )
  
    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None, flag=None, enc=True):
        layers = []
        if pool_kernel is not None:
            if pool_kernel > 0:
                layers.append(nn.AvgPool2d(pool_kernel))
            elif pool_kernel < 0:
                layers.append(nn.UpsamplingNearest2d(scale_factor=-pool_kernel))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) // 2))
        layers.append(nn.BatchNorm2d(out_ch))
        # Dropout
        if flag is not None:
            layers.append(nn.Dropout2d(0.5))
        # LeakyReLU or ReLU
        if enc is True:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif enc is False:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
  
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        out = self.dec1(x4)
        out = self.dec2(torch.cat([out, x3], dim=1))
        out = self.dec3(torch.cat([out, x2], dim=1))
        out = self.dec4(torch.cat([out, x1], dim=1))
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_bn_relu(4, 16, kernel_size=5, reps=1) # fake/true opt+sar
        self.conv2 = self.conv_bn_relu(16, 32, pool_kernel=4)
        self.conv3 = self.conv_bn_relu(32, 64, pool_kernel=2)
        self.out_patch = nn.Conv2d(64, 1, kernel_size=1)

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None, reps=2):
        layers = []
        for i in range(reps):
            if i == 0 and pool_kernel is not None:
                layers.append(nn.AvgPool2d(pool_kernel))
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch,
                                  out_ch, kernel_size, padding=(kernel_size - 1) // 2))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        return self.out_patch(out)

def train():
    torch.backends.cudnn.benchmark = True
    print(f"Using {device}")
    model_G, model_D = Generator(), Discriminator()
    model_G, model_D = model_G.to(device), model_D.to(device)
    model_G, model_D = nn.DataParallel(model_G), nn.DataParallel(model_D)

    params_G = torch.optim.Adam(model_G.parameters(),lr=0.0002, betas=(0.5, 0.999))
    params_D = torch.optim.Adam(model_D.parameters(),lr=0.0002, betas=(0.5, 0.999))

    # ラベル変数 (PatchGAN),損失関数
    ones = torch.ones(128, 1, 32, 32).to(device)
    zeros = torch.zeros(128, 1, 32, 32).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.L1Loss()

    # 損失を表示するための辞書
    result = {}
    result["log_loss_G_sum"] = []
    result["log_loss_G_bce"] = []
    result["log_loss_G_mae"] = []
    result["log_loss_D"] = []
    
    output_Gsum = []
    output_Gbce = []
    output_Gmae = []
    output_D = []

    # 訓練
    dataset = load_datasets()
    print("Dataset Loaded!!")
    for i in tqdm(range(100), desc="Epochs", unit="epoch"):
        log_loss_G_sum, log_loss_G_bce, log_loss_G_mae, log_loss_D = [], [], [], []

        for (input_gray, real_color) in dataset:
            # input_gray[0] がSAR画像、input_gray[1]がラベル(今回は必要ない)
            # real_color[0]が光学画像、input_gray[1]がラベル            
            batch_len = len(real_color[0])
            real_color, input_gray = real_color[0].to(device), input_gray[0].to(device)
            ### Gの訓練
            # 偽のカラー画像を作成
            fake_color = model_G(input_gray)
            # 識別器の学習の際に生成器に影響が出ないようにするため、偽画像を一時保存
            fake_color_tensor = fake_color.detach()
            # 偽画像を本物と騙せるようにロスを計算
            LAMBD = 100.0 # L1損失と交差エントロピー損失の比率を決める超パラメータ
            out = model_D(torch.cat([fake_color, input_gray], dim=1))
            loss_G_bce = bce_loss(out, ones[:batch_len])
            loss_G_mae = LAMBD * mae_loss(fake_color, real_color)
            loss_G_sum = loss_G_bce + loss_G_mae
            log_loss_G_bce.append(loss_G_bce.item())
            log_loss_G_mae.append(loss_G_mae.item())
            log_loss_G_sum.append(loss_G_sum.item())
            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_G_sum.backward()
            params_G.step()

            ### Discriminatorの訓練
            # 本物のカラー画像を本物と識別できるようにロスを計算
            real_out = model_D(torch.cat([real_color, input_gray], dim=1))
            loss_D_real = bce_loss(real_out, ones[:batch_len])
            # 偽の画像の偽と識別できるようにロスを計算
            fake_out = model_D(torch.cat([fake_color_tensor, input_gray], dim=1))
            loss_D_fake = bce_loss(fake_out, zeros[:batch_len])
            # 実画像と偽画像のロスを合計
            loss_D = loss_D_real + loss_D_fake
            log_loss_D.append(loss_D.item())
            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_D.backward()
            params_D.step()

        result["log_loss_G_sum"].append(statistics.mean(log_loss_G_sum))
        result["log_loss_G_bce"].append(statistics.mean(log_loss_G_bce))
        result["log_loss_G_mae"].append(statistics.mean(log_loss_G_mae))
        result["log_loss_D"].append(statistics.mean(log_loss_D))
        print(f"eposh:{i+1}=>" + f"log_loss_G_sum = {result['log_loss_G_sum'][-1]} " +
              f"({result['log_loss_G_bce'][-1]}, {result['log_loss_G_mae'][-1]}) " +
              f"log_loss_D = {result['log_loss_D'][-1]}")
        
        output_Gsum.append(result['log_loss_G_sum'][-1])
        output_Gbce.append(result['log_loss_G_bce'][-1])
        output_Gmae.append(result['log_loss_G_mae'][-1])
        output_D.append(result['log_loss_D'][-1])
        
        # 画像を保存
        if not os.path.exists("graytopt"):
            os.mkdir("graytopt")
        # 生成画像を保存
        torchvision.utils.save_image(input_gray[:min(batch_len, 100)],
                                f"graytopt/gray_epoch_{i:03}.png", normalize=True)
        torchvision.utils.save_image(fake_color_tensor[:min(batch_len, 100)],
                                f"graytopt/fake_epoch_{i:03}.png", normalize=True)
        torchvision.utils.save_image(real_color[:min(batch_len, 100)],
                                f"graytopt/real_epoch_{i:03}.png", normalize=True)

        # 生成器と識別器の学習モデルをそれぞれ保存
        if not os.path.exists("graytopt/models"):
            os.mkdir("graytopt/models")
        if i % 2 == 0 or i == 99:
            torch.save(model_G.state_dict(), f"graytopt/models/gen_{i:03}.pt")                        
            torch.save(model_D.state_dict(), f"graytopt/models/dis_{i:03}.pt") 
            plt.plot(output_Gsum, color = "red")
            plt.plot(output_Gbce, color = "blue")
            plt.plot(output_Gmae, color = "green")
            plt.plot(output_D, color = "black")
            plt.savefig(f"graytopt/models/l_curve_{i:03}.png", dpi=300) 

    
    # ログ
    with open("graytopt/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)
    
    plt.plot(output_Gsum, color = "red")
    plt.plot(output_Gbce, color = "blue")
    plt.plot(output_Gmae, color = "green")
    plt.plot(output_D, color = "black")
    plt.show()

if __name__ == "__main__":
    train()
