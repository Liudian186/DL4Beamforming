import torch
import torch.nn as nn
import torch.nn.functional as F


class Trin(nn.Module):
    def __init__(self):
        super(Trin, self).__init__()

        self.conv1 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)

        # 三输入
        self.input_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.input_conv2 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.input_conv3 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, input1, input2, input3):
        x1 = F.relu(self.input_conv1(input1))
        x2 = F.relu(self.input_conv2(input2))
        x3 = F.relu(self.input_conv3(input3))

        x = torch.cat([x1, x2, x3], dim=1)  # [batch_size, 64*3, 512, 512]
        # print(f"shape: {x.shape}") # 192

        x = F.relu(self.conv1(x))  # 192
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        output = self.deconv3(x)  # (batch_size, 1, 512, 512)

        return output


if __name__ == "__main__":

    model = Trin()

    input1 = torch.randn(1, 1, 512, 512)
    input2 = torch.randn(1, 1, 512, 512)
    input3 = torch.randn(1, 1, 512, 512)

    # 前向传播
    output = model(input1, input2, input3)

    print("Output shape:", output.shape)  # [1, 1, 512, 512]
