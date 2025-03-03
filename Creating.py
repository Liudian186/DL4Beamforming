import torch


def create_weighted_target(TARGET_WEIGHTS, target_img, cf_img, mv_img):
    """
    创建加权目标图像
    """
    weighted_target = (
        TARGET_WEIGHTS["target"] * target_img
        + TARGET_WEIGHTS["cf"] * cf_img
        + TARGET_WEIGHTS["mv"] * mv_img
    )

    # 归一化处理
    min_val = torch.min(weighted_target)
    max_val = torch.max(weighted_target)
    weighted_target = (weighted_target - min_val) / (max_val - min_val)

    # 缩放到原图像的动态范围
    original_min = torch.min(target_img)
    original_max = torch.max(target_img)
    weighted_target = weighted_target * (original_max - original_min) + original_min

    return weighted_target


def iq_to_image(iq_data):
    """将I/Q数据转换为B模式图像"""
    I = iq_data[:, 0, :, :]
    Q = iq_data[:, 1, :, :]
    magnitude = torch.sqrt(I**2 + Q**2)
    if torch.isnan(magnitude).any():
        print("NaN detected in magnitude!")
    return 20 * torch.log10(magnitude + 1e-6)
