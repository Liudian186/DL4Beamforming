import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PlaneWaveData import MYOData, TSHData
from PixelGrid import make_pixel_grid
from das_torch import DAS2img

# 正确显示负号
plt.rcParams["axes.unicode_minus"] = False

# TARGET_SIZE = 512


# 自定义数据集
class MYDataset(Dataset):
    def __init__(
        self,
        zero_RF,
        neg_RF,
        pos_RF,
        zero_img,
        neg_img,
        pos_img,
        target_images,
        grid_shape,
        cf_img,
        mv_img,
    ):
        self.zero_RF = zero_RF
        self.neg_RF = neg_RF
        self.pos_RF = pos_RF
        self.zero_img = zero_img
        self.neg_img = neg_img
        self.pos_img = pos_img
        self.target_images = target_images
        self.grid_shape = grid_shape
        self.cf_img = cf_img
        self.mv_img = mv_img

    def __len__(self):
        return len(self.zero_RF)

    def __getitem__(self, idx):
        return (
            self.zero_RF[idx],
            self.neg_RF[idx],
            self.pos_RF[idx],
            self.zero_img[idx],
            self.neg_img[idx],
            self.pos_img[idx],
            self.target_images[idx],
            self.grid_shape[idx],
            self.cf_img[idx],
            self.mv_img[idx],
        )


def load_data():

    zero_RF = []
    neg_RF = []
    pos_RF = []
    zero_img = []
    neg_img = []
    pos_img = []
    target_images = []
    cf_img = []
    mv_img = []
    grid_shape = []

    # 加载MYO数据集
    for i in range(0):

        database_path = os.path.join("database", "MYO")
        data_idx = i + 1

        temp = MYOData(database_path, data_idx)

        # 生成像素网格
        xlims = [temp.ele_pos[0, 0], temp.ele_pos[-1, 0]]
        zlims = [5e-3, 55e-3]
        wvln = temp.c / temp.fc
        dx = wvln / 3
        dz = dx
        temp_grid = make_pixel_grid(xlims, zlims, dx, dz)

        # 0度IQ信号
        the_zero = len(temp.angles) // 2  # 0度
        idata_temp = temp.idata[the_zero]
        qdata_temp = temp.qdata[the_zero]
        zero_RF.append([idata_temp, qdata_temp])

        # -8度IQ信号
        the_neg = len(temp.angles) // 4  # -8度
        idata_temp = temp.idata[the_neg]
        qdata_temp = temp.qdata[the_neg]
        neg_RF.append([idata_temp, qdata_temp])

        # +8度IQ信号
        the_pos = len(temp.angles) - len(temp.angles) // 4  # +8度
        idata_temp = temp.idata[the_pos]
        qdata_temp = temp.qdata[the_pos]
        pos_RF.append([idata_temp, qdata_temp])

        # 0度target图像
        zero_img.append(DAS2img(temp, temp_grid, range(the_zero - 2, the_zero + 3)))

        # -8度target图像
        neg_img.append(DAS2img(temp, temp_grid, range(the_neg - 2, the_neg + 3)))

        # +8度target图像
        pos_img.append(DAS2img(temp, temp_grid, range(the_pos - 2, the_pos + 3)))

        # 全角度target图像
        target_images.append(DAS2img(temp, temp_grid))

        # 记录像素网格形状
        grid_shape.append(temp_grid.shape[:-1])

    # 加载TSH数据集
    for i in range(1, 11):

        database_path = os.path.join("database", "TSH")
        data_idx = i + 1

        temp = TSHData(database_path, data_idx)

        # 生成像素网格
        xlims = [temp.ele_pos[0, 0], temp.ele_pos[-1, 0]]
        zlims = [10e-3, 45e-3]
        wvln = temp.c / temp.fc
        dx = wvln / 2.5
        dz = dx
        temp_grid = make_pixel_grid(xlims, zlims, dx, dz)

        # 0度IQ信号
        the_zero = len(temp.angles) // 2
        idata_temp = temp.idata[the_zero]
        qdata_temp = temp.qdata[the_zero]
        zero_RF.append([idata_temp, qdata_temp])

        # -8度IQ信号
        the_neg = len(temp.angles) // 4
        idata_temp = temp.idata[the_neg]
        qdata_temp = temp.qdata[the_neg]
        neg_RF.append([idata_temp, qdata_temp])

        # +8度IQ信号
        the_pos = len(temp.angles) - len(temp.angles) // 4
        idata_temp = temp.idata[the_pos]
        qdata_temp = temp.qdata[the_pos]
        pos_RF.append([idata_temp, qdata_temp])

        # 0度target图像
        zero_img.append(DAS2img(temp, temp_grid, range(the_zero - 2, the_zero + 3)))

        # -8度target图像
        neg_img.append(DAS2img(temp, temp_grid, range(the_neg - 2, the_neg + 3)))

        # +8度target图像
        pos_img.append(DAS2img(temp, temp_grid, range(the_pos - 2, the_pos + 3)))

        # 全角度target图像
        target_images.append(DAS2img(temp, temp_grid))

        # 全角度cf图像
        cf_filename = "TSH" + str(data_idx).zfill(3) + "CFFull.csv"
        cf_path = os.path.join(database_path, cf_filename)
        cf_data = pd.read_csv(cf_path, header=None, skiprows=1)
        cf_tensor = torch.tensor(cf_data.values, dtype=torch.float32)
        cf_img.append(cf_tensor)

        # 全角度mv图像
        mv_filename = "TSH" + str(data_idx).zfill(3) + "MVFull.csv"
        mv_path = os.path.join(database_path, mv_filename)
        mv_data = pd.read_csv(mv_path, header=None, skiprows=1)
        mv_tensor = torch.tensor(mv_data.values, dtype=torch.float32)
        mv_img.append(mv_tensor)

        # 记录像素网格形状
        grid_shape.append(temp_grid.shape[:-1])

    # 处理zero_img
    for i in range(len(zero_img)):
        mask0 = ~torch.isfinite(zero_img[i])
        zero_img[i] = zero_img[i].masked_fill(mask0, 0)

    # 处理neg_img
    for i in range(len(neg_img)):
        mask0 = ~torch.isfinite(neg_img[i])
        neg_img[i] = neg_img[i].masked_fill(mask0, 0)

    # 处理pos_img
    for i in range(len(pos_img)):
        mask0 = ~torch.isfinite(pos_img[i])
        pos_img[i] = pos_img[i].masked_fill(mask0, 0)

    # 处理target_images
    for i in range(len(target_images)):
        mask0 = ~torch.isfinite(target_images[i])
        target_images[i] = target_images[i].masked_fill(mask0, 0)

    zero_RF = torch.tensor(zero_RF, dtype=torch.float32)
    neg_RF = torch.tensor(neg_RF, dtype=torch.float32)
    pos_RF = torch.tensor(pos_RF, dtype=torch.float32)
    zero_img = torch.stack(zero_img)
    neg_img = torch.stack(neg_img)
    pos_img = torch.stack(pos_img)
    target_images = torch.stack(target_images)
    grid_shape = torch.tensor(grid_shape, dtype=torch.int32)

    return (
        zero_RF,
        neg_RF,
        pos_RF,
        zero_img,
        neg_img,
        pos_img,
        target_images,
        grid_shape,
        cf_img,
        mv_img,
    )


if __name__ == "__main__":

    (
        zero_RF,
        neg_RF,
        pos_RF,
        zero_img,
        neg_img,
        pos_img,
        target_images,
        grid_shape,
        cf_img,
        mv_img,
    ) = load_data()

    mydata = MYDataset(
        zero_RF,
        neg_RF,
        pos_RF,
        zero_img,
        neg_img,
        pos_img,
        target_images,
        grid_shape,
        cf_img,
        mv_img,
    )

    mydata_loader = torch.utils.data.DataLoader(mydata, batch_size=1)

    for i, batch in enumerate(mydata_loader):
        (
            zero_RF,
            neg_RF,
            pos_RF,
            zero_img,
            neg_img,
            pos_img,
            target_images,
            grid_shape,
            cf_img,
            mv_img,
        ) = batch

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 展示0度图像
        im0 = axes[0, 0].imshow(zero_img[0].cpu(), cmap="gray")
        axes[0, 0].set_title("0度图像")
        plt.colorbar(im0, ax=axes[0, 0])

        # 展示0度RF信号
        im1 = axes[0, 1].imshow(neg_img[0].cpu(), cmap="gray")
        axes[0, 1].set_title("-8度图像")
        plt.colorbar(im1, ax=axes[0, 1])

        # 展示-8度图像
        im2 = axes[1, 0].imshow(pos_img[0].cpu(), cmap="gray")
        axes[1, 0].set_title("+8度图像")
        plt.colorbar(im2, ax=axes[1, 0])

        # 展示-8度RF信号
        im2 = axes[1, 1].imshow(target_images[0].cpu(), cmap="gray")
        axes[1, 1].set_title("75角度图像")
        plt.colorbar(im2, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

    print("数据加载完成！")
