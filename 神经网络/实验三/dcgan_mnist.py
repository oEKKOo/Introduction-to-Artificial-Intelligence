import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

# ============ 1. 环境与超参数 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据及网络相关超参数
batch_size = 128       # 批大小
lr = 2e-4              # 学习率
num_epochs = 50        # 训练轮数
latent_dim = 100       # 生成器输入噪声向量维度
ngf = 64               # 生成器中 feature map 基数
ndf = 64               # 判别器中 feature map 基数

# 生成图像的输出文件夹
sample_dir = 'samples_dcgan_32x32'
os.makedirs(sample_dir, exist_ok=True)

# ============ 2. 数据准备：MNIST → 32x32 ============

transform = transforms.Compose([
    transforms.Resize(32),          # 将 28×28 缩放到 32×32
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, +1]
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# ============ 3. 定义生成器 (G) ============
#  目标：输入 (N, latent_dim, 1, 1) -> 输出 (N, 1, 32, 32)
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            # (nz, 1, 1) -> (ngf*4, 4, 4)
            nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4, 4, 4) -> (ngf*2, 8, 8)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2, 8, 8) -> (ngf, 16, 16)
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf, 16, 16) -> (1, 32, 32)
            nn.ConvTranspose2d(ngf, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# ============ 4. 定义判别器 (D) ============
#  目标：输入 (N, 1, 32, 32) -> 输出 (N, 1)
#  卷积层逐步下采样到 1×1
class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # (1, 32, 32) -> (ndf, 16, 16)
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf, 16, 16) -> (ndf*2, 8, 8)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2, 8, 8) -> (ndf*4, 4, 4)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4, 4, 4) -> (ndf*8, 2, 2)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8, 2, 2) -> (1, 1, 1)
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ============ 5. 实例化网络 ============

G = Generator(nz=latent_dim, ngf=ngf).to(device)
D = Discriminator(ndf=ndf).to(device)

# ============ 6. 损失函数 & 优化器 ============

criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# ============ 7. 训练循环 ============

def denorm(x):
    """ 将 [-1,1] 的生成结果映射回 [0,1] 用于可视化 """
    out = (x + 1) / 2
    return out.clamp(0, 1)

fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)  # 固定噪声，观测训练进度
step = 0

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        bs = images.size(0)
        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        # ---------------------
        # (1) 训练判别器
        # ---------------------
        images = images.to(device)
        # 判别器对真实图的结果
        outputs = D(images).view(-1, 1)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs.mean().item()

        # 判别器对假图的结果
        z = torch.randn(bs, latent_dim, 1, 1, device=device)
        fake_images = G(z)
        outputs = D(fake_images.detach()).view(-1, 1)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs.mean().item()

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ---------------------
        # (2) 训练生成器
        # ---------------------
        # 让生成器骗过判别器 => 输出应为 real_labels
        outputs = D(fake_images).view(-1, 1)
        g_loss = criterion(outputs, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        step += 1
        if (i+1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                  f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, "
                  f"D(x): {real_score:.2f}, D(G(z)): {fake_score:.2f}")

    # 每个 epoch 结束后，用固定噪声生成一批图像并保存
    G.eval()
    with torch.no_grad():
        fake = G(fixed_noise).cpu()
    G.train()

    fake = denorm(fake)
    save_image(fake, os.path.join(sample_dir, f'fake_images_epoch_{epoch+1:03d}.png'), nrow=8)

print("Training finished. Saving final model...")
torch.save(G.state_dict(), "generator_dcgan_32x32.pth")
torch.save(D.state_dict(), "discriminator_dcgan_32x32.pth")
