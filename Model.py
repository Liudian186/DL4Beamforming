import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理可能的尺寸差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SubNetwork(nn.Module):
    """处理单个角度的子网络（结构与示例UNet相同）"""

    def __init__(self, in_channels=2, base_channels=64):
        super().__init__()
        # 复用示例中的UNet结构
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, 2, kernel_size=1)  # 固定输出2通道

    def forward(self, x, target_h, target_w):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出层与尺寸调整
        out = self.outc(x)
        return F.interpolate(
            out, (target_h, target_w), mode="bilinear", align_corners=False
        )


class FusionUNet(nn.Module):
    """多角度融合网络"""

    def __init__(self, in_channels=6, base_channels=64):  # 3个角度×2通道=6
        super().__init__()
        # 修改后的UNet结构
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x, target_h, target_w):
        # 编码-解码流程
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 最终输出
        out = self.outc(x)
        return F.interpolate(
            out, (target_h, target_w), mode="bilinear", align_corners=False
        )


class BeamformingModel(nn.Module):
    """完整模型架构"""

    def __init__(self, base_channels=64):
        super().__init__()
        self.subnet_neg8 = SubNetwork()
        self.subnet_0 = SubNetwork()
        self.subnet_pos8 = SubNetwork()
        self.fusion_net = FusionUNet()

    def forward(self, inputs, target_size):
        """
        Args:
            inputs: 包含三个角度的输入数据元组 (neg8, zero, pos8)
                    每个张量形状: [B, 2, H, W]
            target_size: 目标输出尺寸 (H, W)
        Returns:
            final_output: 最终输出 [B, 2, target_H, target_W]
            subnet_outputs: 子网络输出列表
        """
        target_h, target_w = target_size

        # 各子网络处理
        out_neg8 = self.subnet_neg8(inputs[0], target_h, target_w)
        out_0 = self.subnet_0(inputs[1], target_h, target_w)
        out_pos8 = self.subnet_pos8(inputs[2], target_h, target_w)

        # 通道维度拼接 (B, 6, H, W)
        combined = torch.cat([out_neg8, out_0, out_pos8], dim=1)

        # 融合处理
        final_output = self.fusion_net(combined, target_h, target_w)

        return final_output, [out_neg8, out_0, out_pos8]


if __name__ == "__main__":
    # 配置参数
    batch_size = 1
    input_channels = 2
    input_size = (256, 256)
    target_size = (512, 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = BeamformingModel().to(device)
    print(f"Model structure:\n{model}")

    # 创建模拟输入数据
    input_neg8 = torch.randn(batch_size, input_channels, *input_size).to(device)
    input_0 = torch.randn(batch_size, input_channels, *input_size).to(device)
    input_pos8 = torch.randn(batch_size, input_channels, *input_size).to(device)

    # 前向传播测试
    model.eval()
    with torch.no_grad():
        final_out, subnet_outs = model(
            inputs=(input_neg8, input_0, input_pos8), target_size=target_size
        )

    # 验证输出尺寸
    print("\nOutput verification:")
    print(
        f"Final output shape: {final_out.shape} (expected: [{batch_size}, 2, {target_size[0]}, {target_size[1]}])"
    )

    for i, out in enumerate(subnet_outs):
        print(
            f"Subnet {i} output shape: {out.shape} (expected: [{batch_size}, 2, {target_size[0]}, {target_size[1]}])"
        )

    # 自动化断言检查
    assert final_out.shape == (batch_size, 2, *target_size), "Final output尺寸错误"
    for out in subnet_outs:
        assert out.shape == (batch_size, 2, *target_size), "子网络输出尺寸错误"
    print("\n所有输出尺寸验证通过！")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params/1e6:.2f}M")
    print("模型参数量统计完成！")
