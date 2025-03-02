import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from LoadData import MYDataset, load_data
from Model import BeamformingModel  # 确保模型包含子网络输出

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 600

# 加载数据
(
    zero_RF, neg_RF, pos_RF,
    zero_img, neg_img, pos_img,
    target_img, grid_shape
) = load_data()

# 创建数据集
dataset = MYDataset(
    zero_RF, neg_RF, pos_RF,
    zero_img, neg_img, pos_img,
    target_img, grid_shape
)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# 初始化模型和优化器
model = BeamformingModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 定义子网络权重
subnet_weights = [0.2, 0.2, 0.2]  # 三个子网络权重
final_weight = 0.4                # 最终输出权重

train_losses = []
val_losses = []

def iq_to_image(iq_data):
    """将I/Q数据转换为B模式图像"""
    I = iq_data[:, 0, :, :]
    Q = iq_data[:, 1, :, :]
    magnitude = torch.sqrt(I**2 + Q**2)
    if torch.isnan(magnitude).any():
        print("NaN detected in magnitude!")
    return 20 * torch.log10(magnitude + 1e-6)

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for batch in train_loader:
        # 解包数据
        (zero_RF, neg_RF, pos_RF,
         zero_img, neg_img, pos_img,
         target_img, grid_shape) = batch
        
        # 转移数据到设备
        inputs = [
            zero_RF.to(device),
            neg_RF.to(device),
            pos_RF.to(device)
        ]
        target_imgs = {
            'zero': zero_img.to(device),
            'neg': neg_img.to(device),
            'pos': pos_img.to(device),
            'final': target_img.to(device)
        }
        target_h, target_w = grid_shape[0, 0], grid_shape[0, 1]

        # 前向传播
        optimizer.zero_grad()
        final_output, subnet_outputs = model(inputs, (target_h, target_w))  # 获取子网络输出
        
        # 计算各子网络损失
        subnet_losses = []
        for subnet_out, img_key in zip(subnet_outputs, ['zero', 'neg', 'pos']):
            subnet_img = iq_to_image(subnet_out)
            subnet_img = subnet_img - torch.amax(subnet_img)
            loss = criterion(subnet_img, target_imgs[img_key])
            subnet_losses.append(loss)
        
        # 计算最终输出损失
        final_img = iq_to_image(final_output)
        final_img = final_img - torch.amax(final_img)
        final_loss = criterion(final_img, target_imgs['final'])
        
        # 总损失加权求和
        total_loss = final_weight * final_loss
        for w, l in zip(subnet_weights, subnet_losses):
            total_loss += w * l
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()

    # 记录平均训练损失
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            (zero_RF, neg_RF, pos_RF,
             zero_img, neg_img, pos_img,
             target_img, grid_shape) = batch
            
            inputs = [
                zero_RF.to(device),
                neg_RF.to(device),
                pos_RF.to(device)
            ]
            target_imgs = {
                'zero': zero_img.to(device),
                'neg': neg_img.to(device),
                'pos': pos_img.to(device),
                'final': target_img.to(device)
            }
            target_h, target_w = grid_shape[0, 0], grid_shape[0, 1]
            
            final_output, subnet_outputs = model(inputs, (target_h, target_w))
            
            # 子网络损失
            subnet_losses = []
            for subnet_out, img_key in zip(subnet_outputs, ['zero', 'neg', 'pos']):
                subnet_img = iq_to_image(subnet_out)
                subnet_img = subnet_img - torch.amax(subnet_img)
                loss = criterion(subnet_img, target_imgs[img_key])
                subnet_losses.append(loss)
            
            # 最终损失
            final_img = iq_to_image(final_output)
            final_img = final_img - torch.amax(final_img)
            final_loss = criterion(final_img, target_imgs['final'])
            
            # 总损失
            total_val_loss = final_weight * final_loss
            for w, l in zip(subnet_weights, subnet_losses):
                total_val_loss += w * l
            
            val_loss += total_val_loss.item()
    
    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    model.train()
    
    print(f'Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

# 损失曲线绘制 (与原始代码相同)
plt.figure()
plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, epochs+1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("train_val_loss.png")
plt.close()

# 对比图像生成函数（增加子网络显示）
def generate_comparison_images(loader, dataset_type):
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))  # 增加列数显示子网络结果
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            (zero_RF, neg_RF, pos_RF,
             zero_img, neg_img, pos_img,
             target_img, grid_shape) = batch
            
            inputs = [
                zero_RF.to(device),
                neg_RF.to(device),
                pos_RF.to(device)
            ]
            target_h, target_w = grid_shape[0, 0].item(), grid_shape[0, 1].item()
            
            final_output, subnet_outputs = model(inputs, (target_h, target_w))
            
            # 生成各子网络图像
            subnet_imgs = []
            for subnet_out in subnet_outputs:
                img = iq_to_image(subnet_out)
                img = img - torch.amax(img)
                subnet_imgs.append(img.cpu().numpy())
            
            # 生成最终图像
            final_img = iq_to_image(final_output)
            final_img = final_img - torch.amax(final_img)
            final_img = final_img.cpu().numpy()
            
            # 绘制结果
            axes[i, 0].imshow(target_img[0].squeeze().cpu().numpy(), cmap="gray")
            axes[i, 0].set_title("Target")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(subnet_imgs[0][0], cmap="gray")
            axes[i, 1].set_title("Zero Subnet")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(subnet_imgs[1][0], cmap="gray")
            axes[i, 2].set_title("Neg Subnet")
            axes[i, 2].axis("off")
            
            axes[i, 3].imshow(subnet_imgs[2][0], cmap="gray")
            axes[i, 3].set_title("Pos Subnet")
            axes[i, 3].axis("off")

            axes[i, 4].imshow(final_img[0], cmap="gray")
            axes[i, 4].set_title("Final Output")
            axes[i, 4].axis("off")
    
    plt.savefig(f"{dataset_type}_comparison.png", bbox_inches="tight")
    plt.close()

generate_comparison_images(train_loader, "train")
generate_comparison_images(val_loader, "val")

print("Training completed!")