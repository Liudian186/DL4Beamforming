import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from LoadData import MYDataset, load_data
from Model import BeamformingModel  # 确保模型包含子网络输出
from Creating import create_weighted_target, iq_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 500
mid = 30

# 加载数据
(
    zero_RF,
    neg_RF,
    pos_RF,
    zero_img,
    neg_img,
    pos_img,
    target_img,
    grid_shape,
    cf_img,
    mv_img,
) = load_data()

# 创建数据集
dataset = MYDataset(
    zero_RF,
    neg_RF,
    pos_RF,
    zero_img,
    neg_img,
    pos_img,
    target_img,
    grid_shape,
    cf_img,
    mv_img,
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
final_weight = 0.4  # 最终输出权重

# 在文件开头添加新的权重定义
final_weights = {"target": 0.4, "cf": 0.3, "mv": 0.3}

train_losses = []
val_losses = []

# 定义损失记录字典
losses = {
    "stage1": {"subnet1": [], "subnet2": [], "subnet3": []},
    "stage2": {
        "subnet1": [],
        "subnet2": [],
        "subnet3": [],
        "fusion": {
            "train": {"target": [], "cf": [], "mv": [], "total": []},
            "val": {"target": [], "cf": [], "mv": [], "total": []},
        },
    },
}

SUBNET_WEIGHTS = [0.2, 0.2, 0.2]  # 子网络权重
FINAL_WEIGHTS = {
    "target": 0.4,  # 目标图像权重
    "cf": 0.3,  # CF图像权重
    "mv": 0.3,  # MV图像权重
}

# 添加固定权重常量
TARGET_WEIGHTS = {"target": 0.4, "cf": 0.3, "mv": 0.3}


# 冻结融合网络的参数
for param in model.fusion_net.parameters():
    param.requires_grad = False

# 第一阶段: 只训练子网络
print("Stage 1: Training subnets only...")
for epoch in range(mid):
    subnet_epoch_losses = [0.0, 0.0, 0.0]
    running_loss = 0.0
    for batch in train_loader:
        # 解包数据
        (
            zero_RF,
            neg_RF,
            pos_RF,
            zero_img,
            neg_img,
            pos_img,
            target_img,
            grid_shape,
            cf_img,
            mv_img,
        ) = batch

        inputs = [zero_RF.to(device), neg_RF.to(device), pos_RF.to(device)]
        target_imgs = {
            "zero": zero_img.to(device),
            "neg": neg_img.to(device),
            "pos": pos_img.to(device),
            "final": target_img.to(device),
            "cf": cf_img.to(device),
            "mv": mv_img.to(device),
        }
        target_h, target_w = grid_shape[0, 0], grid_shape[0, 1]

        optimizer.zero_grad()

        # 前向传播
        final_output, subnet_outputs = model(inputs, (target_h, target_w))

        # 只计算子网络损失
        subnet_losses = []
        for subnet_out, img_key in zip(subnet_outputs, ["zero", "neg", "pos"]):
            subnet_img = iq_to_image(subnet_out)
            subnet_img = subnet_img - torch.amax(subnet_img)
            loss = criterion(subnet_img, target_imgs[img_key])
            subnet_losses.append(loss)

        # 总损失只包含子网络损失
        total_loss = sum([w * l for w, l in zip(subnet_weights, subnet_losses)])

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        # 记录每个子网络的损失
        for i, (subnet_loss, img_key) in enumerate(
            zip(subnet_losses, ["zero", "neg", "pos"])
        ):
            subnet_epoch_losses[i] += subnet_loss.item()

    # 计算平均损失并存储
    for i in range(3):
        avg_loss = subnet_epoch_losses[i] / len(train_loader)
        losses["stage1"][f"subnet{i+1}"].append(avg_loss)

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    print(f"Stage 1 - Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f}")

# 解冻融合网络
print("Stage 2: Training full model...")
for param in model.fusion_net.parameters():
    param.requires_grad = True

# 重新初始化优化器以包含所有参数
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 第二阶段: 训练整个网络
for epoch in range(mid, epochs):
    subnet_epoch_losses = [0.0, 0.0, 0.0]
    fusion_epoch_loss = 0.0
    running_loss = 0.0
    for batch in train_loader:
        # 解包数据
        (
            zero_RF,
            neg_RF,
            pos_RF,
            zero_img,
            neg_img,
            pos_img,
            target_img,
            grid_shape,
            cf_img,
            mv_img,
        ) = batch

        inputs = [zero_RF.to(device), neg_RF.to(device), pos_RF.to(device)]
        target_imgs = {
            "zero": zero_img.to(device),
            "neg": neg_img.to(device),
            "pos": pos_img.to(device),
            "final": target_img.to(device),
            "cf": cf_img.to(device),
            "mv": mv_img.to(device),
        }
        target_h, target_w = grid_shape[0, 0], grid_shape[0, 1]

        optimizer.zero_grad()

        final_output, subnet_outputs = model(inputs, (target_h, target_w))

        # 计算子网络损失
        subnet_losses = []
        for subnet_out, img_key in zip(subnet_outputs, ["zero", "neg", "pos"]):
            subnet_img = iq_to_image(subnet_out)
            subnet_img = subnet_img - torch.amax(subnet_img)
            loss = criterion(subnet_img, target_imgs[img_key])
            subnet_losses.append(loss)

        # 计算最终输出损失
        final_img = iq_to_image(final_output)
        final_img = final_img - torch.amax(final_img)

        # 创建加权目标图像
        weighted_target = create_weighted_target(
            TARGET_WEIGHTS, target_imgs["final"], target_imgs["cf"], target_imgs["mv"]
        )

        # 计算损失
        final_loss = criterion(final_img, weighted_target)

        # 总损失计算保持不变:
        total_loss = final_loss
        for w, l in zip(SUBNET_WEIGHTS, subnet_losses):
            total_loss += w * l

        # 添加retain_graph=True
        total_loss.backward(retain_graph=True)
        optimizer.step()

        running_loss += total_loss.item()

        # 记录损失
        for i, subnet_loss in enumerate(subnet_losses):
            subnet_epoch_losses[i] += subnet_loss.item()
        fusion_epoch_loss += final_loss.item()

    # 计算平均损失并存储
    for i in range(3):
        avg_loss = subnet_epoch_losses[i] / len(train_loader)
        losses["stage2"][f"subnet{i+1}"].append(avg_loss)

    # 移除原来的多目标损失记录
    losses["stage2"]["fusion"]["train"]["total"].append(final_loss.item())

    # 验证集评估
    model.eval()
    val_fusion_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            # 解包数据
            (
                zero_RF,
                neg_RF,
                pos_RF,
                zero_img,
                neg_img,
                pos_img,
                target_img,
                grid_shape,
                cf_img,
                mv_img,
            ) = batch

            inputs = [zero_RF.to(device), neg_RF.to(device), pos_RF.to(device)]
            target_imgs = {
                "zero": zero_img.to(device),
                "neg": neg_img.to(device),
                "pos": pos_img.to(device),
                "final": target_img.to(device),
                "cf": cf_img.to(device),
                "mv": mv_img.to(device),
            }
            target_h, target_w = grid_shape[0, 0], grid_shape[0, 1]

            final_output, _ = model(inputs, (target_h, target_w))
            final_img = iq_to_image(final_output)
            final_img = final_img - torch.amax(final_img)

            # 创建加权目标图像
            weighted_target = create_weighted_target(
                TARGET_WEIGHTS,
                target_imgs["final"],
                target_imgs["cf"],
                target_imgs["mv"],
            )

            # 计算验证损失
            val_loss = criterion(final_img, weighted_target)
            val_fusion_loss += val_loss.item()

    avg_val_loss = val_fusion_loss / len(val_loader)
    losses["stage2"]["fusion"]["val"]["total"].append(avg_val_loss)

    # 移除不需要的验证损失记录
    """
    losses["stage2"]["fusion"]["val"]["target"].append(final_losses["target"].item())
    losses["stage2"]["fusion"]["val"]["cf"].append(final_losses["cf"].item())
    losses["stage2"]["fusion"]["val"]["mv"].append(final_losses["mv"].item())
    """

    model.train()

    # 记录训练损失
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    print(f"Stage 2 - Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f}")

# 验证阶段
model.eval()
val_loss = 0.0
with torch.no_grad():
    for batch in val_loader:
        (
            zero_RF,
            neg_RF,
            pos_RF,
            zero_img,
            neg_img,
            pos_img,
            target_img,
            grid_shape,
            cf_img,
            mv_img,
        ) = batch

        inputs = [zero_RF.to(device), neg_RF.to(device), pos_RF.to(device)]
        target_imgs = {
            "zero": zero_img.to(device),
            "neg": neg_img.to(device),
            "pos": pos_img.to(device),
            "final": target_img.to(device),
            "cf": cf_img.to(device),
            "mv": mv_img.to(device),
        }
        target_h, target_w = grid_shape[0, 0], grid_shape[0, 1]

        final_output, subnet_outputs = model(inputs, (target_h, target_w))

        # 子网络损失
        subnet_losses = []
        for subnet_out, img_key in zip(subnet_outputs, ["zero", "neg", "pos"]):
            subnet_img = iq_to_image(subnet_out)
            subnet_img = subnet_img - torch.amax(subnet_img)
            loss = criterion(subnet_img, target_imgs[img_key])
            subnet_losses.append(loss)

        # 最终损失
        final_img = iq_to_image(final_output)
        final_img = final_img - torch.amax(final_img)
        final_loss = criterion(final_img, target_imgs["final"])

        # 总损失
        total_val_loss = final_weight * final_loss
        for w, l in zip(subnet_weights, subnet_losses):
            total_val_loss += w * l

        val_loss += total_val_loss.item()

val_loss = val_loss / len(val_loader)
val_losses.append(val_loss)
model.train()

print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# 绘制训练损失曲线
plt.figure(figsize=(15, 5))

# 第一阶段损失曲线
plt.subplot(1, 2, 1)
for i in range(3):
    plt.plot(range(mid), losses["stage1"][f"subnet{i+1}"], label=f"Subnet {i+1}")
plt.title("Stage 1: Subnet Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 第二阶段损失曲线
plt.subplot(1, 2, 2)
for i in range(3):
    plt.plot(
        range(mid, epochs), losses["stage2"][f"subnet{i+1}"], label=f"Subnet {i+1}"
    )
plt.plot(
    range(mid, epochs),
    losses["stage2"]["fusion"]["train"]["total"],
    label="Fusion Net (Train)",
    linestyle="--",
)
plt.plot(
    range(mid, epochs),
    losses["stage2"]["fusion"]["val"]["total"],
    label="Fusion Net (Val)",
    linestyle=":",
)
plt.title("Stage 2: Full Model Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()


# 对比图像生成函数（增加子网络显示）
def generate_comparison_images(loader, dataset_type):
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))  # 增加列数显示子网络结果
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            (
                zero_RF,
                neg_RF,
                pos_RF,
                zero_img,
                neg_img,
                pos_img,
                target_img,
                grid_shape,
                cf_img,
                mv_img,
            ) = batch

            inputs = [zero_RF.to(device), neg_RF.to(device), pos_RF.to(device)]
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
