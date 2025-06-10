import sys
sys.path.append("/home/laicy/process_data/process_CT/functions")

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from os.path import join,exists
from os import makedirs,getcwd,listdir
import numpy as np
from plt_ct import show_box
import torch
import random
import torchvision.transforms.functional as F
from torchvision import transforms
import cv2
import re
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd
)
import imageio.v3 as imageio

def get_ct_names(ct_path):
    #获取ct的名字，返回的是CT名的列表
    #ct_path：ct名字的列表
    #ct_names:返回的是ct名字的列表

    nums=listdir(ct_path)
    ct_names=[]
    for num in nums:
        match = re.match(r"(\d+-\d+-Pre)",num)
        if match:
            ct_names.append(match.group(1))
    return list(set(ct_names))

def standardize_ct_per_slice(ct_image,tumor_range=(-1000, 50), contrast_factor=2.0):
    """
    对每一张CT切片进行标准化处理。

    参数:
        ct_image (numpy.ndarray): 形状为 (D,H, W) 的3D CT图像。
    
    返回:
        numpy.ndarray: 标准化后的CT图像。
    """
    standardize_slices=[]
    min_value = np.min(ct_image)
    max_value = np.max(ct_image)
    for slice_idx in range(ct_image.shape[0]):
        normalized_image = (ct_image[slice_idx,:,:] - min_value) / (max_value - min_value) * 255
        normalized_image = np.uint8(normalized_image)
        standardize_slices.append(normalized_image)
    return np.stack(standardize_slices, axis=0)

def max_min_normalize(img):
    """
    对输入图像进行 Max-Min 归一化。

    参数:
    - img: numpy 数组，形状为 [H, W, C]。

    返回:
    - img_normalized: numpy 数组，归一化后的图像，形状为 [H, W, C]。
    """
    img_min = np.min(img)
    img_max = np.max(img)
    img_normalized = (img - img_min) / (img_max - img_min)
    return img_normalized

def aug(img_np):
    # 将 (H, W, C) 转换为 (C, H, W)
    img_np = np.transpose(img_np, (2, 0, 1))  # 从 (224, 224, n) 转换为 (n, 224, 224)

    # 将 NumPy 数组转换为 PyTorch 张量
    img_tensor = torch.from_numpy(img_np).float() # 归一化到 [0, 1]

    # 随机反转
    if random.random() > 0.5:
        img_tensor = F.hflip(img_tensor)  # 水平翻转
    if random.random() > 0.5:
        img_tensor = F.vflip(img_tensor)  # 垂直翻转

    # 随机旋转
    angle = random.uniform(-30, 30)  # 随机角度
    img_tensor = F.rotate(img_tensor, angle)  # 旋转

    # 随机裁剪
    crop_size = (60, 60)
    i, j, h, w = transforms.RandomCrop.get_params(img_tensor, output_size=crop_size)  # 随机裁剪参数
    img_tensor = F.crop(img_tensor, i, j, h, w)  # 裁剪
    img_tensor = F.resize(img_tensor, (224, 224))  # 调整回原始大小

    # # 颜色抖动（单通道图像不需要调整饱和度和色调）
    img_tensor = F.adjust_brightness(img_tensor, random.uniform(0.8, 1.2))  # 亮度调整
    img_tensor = F.adjust_contrast(img_tensor, random.uniform(0.8, 1.2))  # 对比度调整

    # 将 PyTorch 张量转换回 NumPy 数组
    img_processed_np = img_tensor.numpy()  # 转换为 NumPy 数组
    img_processed_np = np.transpose(img_processed_np, (1, 2, 0))  # 从 (C, H, W) 转换为 (H, W, C) 方便画图

    # 归一化
    img_processed_np = max_min_normalize(img_processed_np)
    # print("aug:",img_processed_np.shape)
    return img_processed_np

def resample_image(image, mask, target_spacing=[1.0, 1.0, 1.0]):
    """
    对 CT 图像和 mask 进行重采样
    :param image: CT 图像 (SimpleITK 图像对象)
    :param mask: mask 图像 (SimpleITK 图像对象)
    :param target_spacing: 目标 spacing，默认为 [1.0, 1.0, 1.0]
    :return: 重采样后的 CT 图像和 mask (SimpleITK 图像对象)
    """
    # 获取原始 spacing 和 size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 计算重采样后的 size
    target_size = [
        int(np.round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]

    # 创建重采样滤波器
    resampler = sitk.ResampleImageFilter()

    # 设置重采样参数
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    # 对 CT 图像进行重采样（使用线性插值）
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_image = resampler.Execute(image)

    # 对 mask 进行重采样（使用最近邻插值）
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask = resampler.Execute(mask)

    return resampled_image, resampled_mask

def load_nii(ct_path, ct_mask_path, num, resample=False, plot=False, enhance=True,
             window_width=350, window_level=50):
    """
    加载并处理 .nii 文件，转换为 NumPy 数组，支持增强和可视化。

    参数:
        ct_path (str): CT 文件路径。
        ct_mask_path (str): Mask 文件路径。
        num (int or str): 用于命名输出文件的标识符。
        resample (bool): 是否进行重采样。
        plot (bool): 是否生成并保存 GIF 可视化。
        enhance (bool): 是否应用 CLAHE 图像增强。
        window_width (int): CT 窗宽。
        window_level (int): CT 窗位。

    返回:
        dict: 包含处理后图像、掩膜、标注帧索引和最大掩膜帧索引的字典。
    """
    # 1. 检查文件存在性
    if not (exists(ct_path) and exists(ct_mask_path)):
        raise FileNotFoundError(f"CT or mask file not found at: {ct_path} or {ct_mask_path}")

    # 2. 读取图像和掩膜
    image = sitk.ReadImage(ct_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(ct_mask_path)

    if resample:
        image, mask = resample_image(image, mask, target_spacing=[1.0, 1.0, 1.0])

    # 4. 将 SimpleITK 对象转换为 NumPy 数组
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    
    # 准备 MONAI 变换的数据字典
    data_dict = {"image": image_array, "mask": mask_array}

    # 5. 定义并应用 MONAI 变换流水线
    # 使用字典版本的变换 ('d' 后缀)
    transforms = Compose([
        # 添加通道维度: (D, H, W) -> (1, D, H, W)
        EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
        # 应用窗宽窗位调整, 将强度映射到 [0, 255]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=window_level - window_width / 2,
            a_max=window_level + window_width / 2,
            b_min=0.0, b_max=255.0, clip=True
        ),
        # 统一方向到 LPS
        Orientationd(keys=["image", "mask"], axcodes="LPS"),
    ])
    
    processed_data = transforms(data_dict)
    
    # 从字典中取出处理后的数组，并移除 MONAI 添加的通道维度以进行后续处理
    # (1, D, H, W) -> (D, H, W)
    image_array = processed_data["image"].squeeze(0).numpy()
    mask_array = processed_data["mask"].squeeze(0).numpy()

    # 6. 尺寸检查
    if image_array.shape != mask_array.shape:
        raise ValueError(f"Size mismatch after transforms: image {image_array.shape}, mask {mask_array.shape}")

    # 7. (可选) 图像增强 (CLAHE)
    if enhance:
        image_array_uint8 = image_array.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_array = np.array([clahe.apply(frame) for frame in image_array_uint8])

    # 8. 标注帧检测
    lagc_indices = [i for i, frame in enumerate(mask_array) if np.any(frame)]
    if not lagc_indices:
        max_frame_idx = mask_array.shape[0] // 2
    else:
        max_frame_idx = lagc_indices[np.argmax([np.count_nonzero(mask_array[i]) for i in lagc_indices])]

    if plot:
        outpath = 'CT_data/groudtruth_gif'
        makedirs(outpath, exist_ok=True)
        frames = []
        # 只为包含掩膜的帧创建 GIF
        for i in lagc_indices:
            # 将灰度图像转换为三通道 RGB
            frame_rgb = np.stack([image_array[i]] * 3, axis=-1).astype(np.uint8)
            # 在掩膜区域上叠加红色
            mask_slice = mask_array[i] > 0
            frame_rgb[mask_slice] = [255, 0, 0] # 将掩膜区域设置为红色
            frames.append(frame_rgb)
        
        if frames:
            imageio.mimsave(join(outpath, f'{num}.gif'), frames, duration=0.17, loop=0)

    # 10. 返回结果
    return {
        'image': image_array.astype(np.float32) / 255.0, # 返回归一化到 [0, 1] 的浮点数图像，更通用
        'mask': mask_array,
        'lagc_indices': lagc_indices,
        'max_frame_idx': max_frame_idx,
    }

def mask_to_box(mask):
    """
    根据二值mask提取bounding box的坐标.
    输入:
        mask: 一个二值化的numpy数组，形状为 (H, W)，值为0或1
    输出:
        (x_min, y_min, x_max, y_max): 包围框的左上角和右下角坐标
    """
    # 找到非零像素点的位置
    y_indices, x_indices = np.where(mask > 0)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有非零像素点，返回一个空框
        return None
    
    # 计算边界
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    box=np.array([x_min,y_min,x_max,y_max],dtype=np.float32)
    return box

def save_to_jpg(num,ct_path,mask_path,outpath,plot_path=None,total=False,plot=True):
    """
    处理CT的数据(.nii),并保存结果
    Args:
        num (_type_): 数据id,str
        ct_path (_type_):CT数据的父文件目录
        mask_path (_type_): mask数据的父文件路径
        outpath (_type_): 
        plot_path (_type_, optional): 保存处理后数据图片的路径. Defaults to None.
        total (bool, optional): 是否保存所有帧数. Defaults to False.
        plot (bool, optional): 是否绘制CT及其mask的gif图片. Defaults to True.

    Raises:
        ValueError: _description_
    """
    # 既返回了box，也返回了点的坐标
    ct_data=load_nii(ct_path,mask_path,num)
    image=ct_data['image']
    mask=ct_data['mask']
    lagc_indices=ct_data['lagc_indices']
    # print(lagc_indices)
    if not total:
        video_array=image[lagc_indices]
        label_array=mask[lagc_indices]
        # 创建文件夹
        image_output_path=join(outpath,num,"video_frame")
        mask_output_path=join(outpath,num,"label")
        makedirs(image_output_path,exist_ok=True)
        makedirs(mask_output_path,exist_ok=True)
        num_frames,_,_=video_array.shape
        for i in range(num_frames):
            if i == 0:
                # 保存点的坐标
                coords=np.column_stack(np.where(label_array[0]==1))
                center=np.mean(coords,axis=0)
                center=center.astype(int)
                np.save(join(mask_output_path,'center_coords.npy'),center)
                # 保存box的坐标
                box=mask_to_box(label_array[0])
                if box is not None:
                    np.save(join(mask_output_path,'box_coords.npy'),box)
                if box is None:
                    print('don\'t get the box coords of CT:',num,i)
                
                
            frame=video_array[i]
            image=Image.fromarray(frame)
            if image.mode !="p":
                image=image.convert('RGB')

            image.save(join(image_output_path,f'{i:03d}.jpg'))
            np.save(join(mask_output_path,f'{i}.npy'),label_array[i])
        if plot:
            if plot_path==None:
                plot_path=join(getcwd(),"true_gif")
                makedirs(plot_path,exist_ok=True)
            elif not isinstance(plot_path, str):
                raise ValueError("plot_path must be a string or None")

            frame=[]
            for i in range(num_frames):
                plt.clf()
                if i ==0:
                    box_plot=mask_to_box(label_array[i])
                    show_box(box_plot,plt.gca())
                cmap = mcolors.ListedColormap(['white', 'red'])
                plt.imshow(video_array[i],cmap='gray')  #vmin=0,vmax=255
                plt.imshow(label_array[i,:,:],cmap=cmap,alpha=0.3)
                plt.savefig('tem.png',bbox_inches='tight',pad_inches=0)
                image=Image.open('tem.png')
                frame.append(image.copy())
                image.close()
            frame[0].save(join(plot_path,f'{num}_truth.gif'),save_all=True,append_images=frame[1:],duration=170,loop=0)

    if total:
        makedirs(join(outpath,f'{num}/video_frame_2/'),exist_ok=True)
        makedirs(join(outpath,f'{num}/label_2/'),exist_ok=True)
        num_frames,_,_=image.shape
        for i in range(num_frames):
            frame=image[i]
            image_=Image.fromarray(frame)
            if image_.mode !="p":
                image_=image_.convert('RGB')
            image_.save(join(outpath,f'{num}/video_frame_2/{i:03d}.jpg'))
            np.save(join(outpath,f'{num}/label_2/{i}.npy'),mask[i])

def save_tumor2jpg_pad(num,ct_path,mask_path,plot=False):
    """提取肿瘤区域,非肿瘤区域填零

    Args:
        num (str): idid
        ct_path (str): root path of ct
        mask_path (str): root path of mask
        plot (bool, optional): 是否画图. Defaults to False.
    """
    # 读取.nii文件
    ct_data=load_nii(ct_path,mask_path,num,resample=False)
    image=ct_data['image']
    mask=ct_data['mask']
    lagc_indices=ct_data['lagc_indices']
    video_array=image[lagc_indices]
    label_array=mask[lagc_indices]

    # 找到mask最大的一帧
    tumor_areas = np.sum(label_array, axis=(1, 2))
    max_tumor_frame_idx = np.argmax(tumor_areas)
    max_tumor_image = video_array[max_tumor_frame_idx]
    max_tumor_mask = label_array[max_tumor_frame_idx]
    # 找到肿瘤区域的边界框
    rows = np.any(max_tumor_mask, axis=1)
    cols = np.any(max_tumor_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 提取肿瘤区域
    tumor_region = np.where(max_tumor_mask > 0, max_tumor_image, 0)
    # 找到肿瘤区域的边界框
    rows = np.any(tumor_region > 0, axis=1)  # 在行方向上是否存在非零值
    cols = np.any(tumor_region > 0, axis=0)  # 在列方向上是否存在非零值
    y_min, y_max = np.where(rows)[0][[0, -1]]  # 行的最小和最大索引
    x_min, x_max = np.where(cols)[0][[0, -1]]  # 列的最小和最大索引

    # 裁剪肿瘤区域并转换成3维图片
    cropped_tumor_region = tumor_region[y_min:y_max+1, x_min:x_max+1]
    
    cropped_tumor_region_3d = np.expand_dims(cropped_tumor_region,axis=0)
    cropped_tumor_region_3d = np.repeat(cropped_tumor_region_3d,3,axis=0).transpose(1,2,0)
    cropped_tumor_region_3d = max_min_normalize(cropped_tumor_region_3d)
    # 对图片进行transforms
    import torchvision.transforms as trans
    trans_ct = trans.Compose([
    trans.ToPILImage(),
    trans.Resize((224, 224)),
    trans.ToTensor(),
])  
      
    trans_tumor_region=trans_ct(cropped_tumor_region_3d)
    trans_tumor_region_np = trans_tumor_region.numpy()
    trans_tumor_region_np = np.transpose(trans_tumor_region_np, (1, 2, 0))
    
    if plot==True:
        aug_img=aug(trans_tumor_region_np)
        plt.figure(figsize=(15, 5))
        # 绘制原始图像
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(max_tumor_image, cmap='gray')  # 原始图像
        plt.axis('off')

        # 绘制原始图像 + Mask
        plt.subplot(2, 3, 2)
        plt.title("Original Image + Mask")
        plt.imshow(max_tumor_image, cmap='gray')  # 原始图像
        plt.imshow(max_tumor_mask, alpha=0.5, cmap='jet')  # 叠加 Mask
        plt.axis('off')

        # 绘制填充后的图像
        plt.subplot(2, 3, 3)
        plt.title("cropped_tumor_region")
        plt.imshow(cropped_tumor_region, cmap='gray')  # 填充后的图像
        plt.axis('off')
        # 绘制transform之后的图片
        plt.subplot(2, 3, 4)
        plt.title("trans_tumor_region_np")
        plt.imshow(trans_tumor_region_np.squeeze(), cmap='gray')  # 变换后的肿瘤区域
        # 3d
        plt.subplot(2, 3, 5)
        plt.title("cropped_tumor_region_3d")
        plt.imshow(cropped_tumor_region_3d.squeeze(), cmap='gray')  # 变换后的肿瘤区域
        # 增强后的数据
        plt.subplot(2,3,6)
        plt.title("after aug")
        plt.imshow(aug_img,cmap="gray")
        plt.axis('off')
        # 显示图像
        plt.tight_layout()
        plt.show()  
        plt.savefig(f"process_CT/images/{num}.png")
    return np.transpose(trans_tumor_region_np, (2, 0, 1))

def norm_tumor(img_arr,mask_arr,plot=True):
    """对肿瘤区域实现标准化(z-score)

    Args:
        img_arr (numpy.darray): 图片矩阵
        mask_arr (numpy.darray): mask

    Returns:
        numpy.darray: 经过标准化之后的肿瘤区域
    """
    img_arr = img_arr * mask_arr

    mean = np.mean(img_arr[img_arr > 0])
    sd = np.std(img_arr[img_arr > 0])

    img_norm_arr = (img_arr - mean) / sd
    img_norm_arr[img_norm_arr < - 5] = - 5
    img_norm_arr[img_norm_arr >   5] = 5
    img_norm_arr = img_norm_arr / 10
    img_norm_arr = img_norm_arr + 0.5
    img_norm_arr = img_norm_arr * mask_arr  #

    if plot:
        normalized_0_1_img=(img_norm_arr-np.min(img_norm_arr))/(np.max(img_norm_arr)-np.min(img_norm_arr))
        plt.imshow(normalized_0_1_img,cmap='gray')
        plt.title("z-score")
        plt.axis('off')
        plt.show()
    return img_norm_arr

def save_tumor2jpg(num,ct_path,mask_path,plot=False):
    """选取肿瘤区域，非旋转的矩形框，选取的就是肿瘤区域的最小矩形区域

    Args:
        num (_type_): ct_name,也就是编号,举例1-05778746-Pre.png
        ct_path (_type_): ct源文件的路径
        mask_path (_type_): mask文件的路径
        plot (bool, optional): 决定是否绘制图像. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        transform之后的图像: （3，224，224）
    """
    # 读取.nii文件
    ct_data=load_nii(ct_path,mask_path,num,resample=False)
    image=ct_data['image']
    mask=ct_data['mask']
    lagc_indices=ct_data['lagc_indices']
    transSIZE=224
    video_array=image[lagc_indices]
    label_array=mask[lagc_indices]

    # 找到mask最大的一帧
    tumor_areas = np.sum(label_array, axis=(1, 2))
    max_tumor_frame_idx = np.argmax(tumor_areas)
    max_tumor_image = video_array[max_tumor_frame_idx]
    max_tumor_mask = label_array[max_tumor_frame_idx]

    # 找到肿瘤区域的边界框
    rows = np.any(max_tumor_mask, axis=1)
    cols = np.any(max_tumor_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 找到肿瘤区域的边界框
    rows = np.any(max_tumor_mask > 0, axis=1)  # 在行方向上是否存在非零值
    cols = np.any(max_tumor_mask > 0, axis=0)  # 在列方向上是否存在非零值
    y_min, y_max = np.where(rows)[0][[0, -1]]  # 行的最小和最大索引
    x_min, x_max = np.where(cols)[0][[0, -1]]  # 列的最小和最大索引

    # 计算最小正方形边界框
    bbox_size = max(y_max - y_min, x_max - x_min)  # 边界框的边长
    if bbox_size %2 ==1:
        bbox_size+=1
    y_center = (y_min + y_max) // 2  # 肿瘤区域的中心 y 坐标
    x_center = (x_min + x_max) // 2  # 肿瘤区域的中心 x 坐标

    # 计算正方形边界框的坐标
    y_start =int(max(0, y_center - 6 - bbox_size // 2))
    y_end = min(max_tumor_image.shape[0], y_center + 6 + bbox_size // 2)
    x_start = int(max(0, x_center - 6 - bbox_size // 2)) 
    x_end = min(max_tumor_image.shape[1], x_center + 6 + bbox_size // 2)

    # 从原始图像上截取正方形区域
    cropped_tumor_region = max_tumor_image[y_start:y_end, x_start:x_end]
    
    cropped_tumor_region_3d = np.expand_dims(cropped_tumor_region,axis=0)
    cropped_tumor_region_3d = np.repeat(cropped_tumor_region_3d,3,axis=0).transpose(1,2,0)

    import torchvision.transforms as trans
    trans_ct = trans.Compose([
    trans.ToPILImage(),
    trans.Resize((transSIZE, transSIZE)),
    trans.ToTensor(),
])  
    # 归一化
    max_,min_=np.max(cropped_tumor_region_3d),np.min(cropped_tumor_region_3d)
    
    if max_==min_==0:
        cropped_tumor_region_3d=cropped_tumor_region_3d
    else:
        cropped_tumor_region_3d=(cropped_tumor_region_3d-min_)/(max_-min_)
    trans_tumor_region=trans_ct(cropped_tumor_region_3d)
    trans_tumor_region_np = trans_tumor_region.numpy()
    trans_tumor_region_np = np.transpose(trans_tumor_region_np, (1, 2, 0))
    if plot==True:
        aug_img=aug(trans_tumor_region_np)
        plt.figure(figsize=(15, 5))
        # 绘制原始图像
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(max_tumor_image, cmap='gray')  # 原始图像
        plt.axis('off')

        # 绘制原始图像 + Mask
        plt.subplot(2, 3, 2)
        plt.title("Original Image + Mask")
        plt.imshow(max_tumor_image, cmap='gray')  # 原始图像
        plt.imshow(max_tumor_mask, alpha=0.5, cmap='jet')  # 叠加 Mask
        plt.axis('off')

        # 绘制填充后的图像
        plt.subplot(2, 3, 3)
        plt.title("cropped_tumor_region")
        plt.imshow(cropped_tumor_region, cmap='gray')  # 填充后的图像
        plt.axis('off')
        # 绘制transform之后的图片
        plt.subplot(2, 3, 4)
        plt.title("Transformed Tumor Region")
        plt.imshow(trans_tumor_region_np.squeeze(), cmap='gray')  # 变换后的肿瘤区域
        # 3d
        plt.subplot(2, 3, 5)
        plt.title("cropped_tumor_region_3d")
        plt.imshow(cropped_tumor_region_3d.squeeze(), cmap='gray')  # 变换后的肿瘤区域
        # 增强后的数据
        plt.subplot(2,3,6)
        plt.title("after aug")
        plt.imshow(aug_img,cmap="gray")
        plt.axis('off')
        # 显示图像
        plt.tight_layout()
        plt.show()  
        plt.savefig(f"process_CT/images/{num}.png")
    return np.transpose(trans_tumor_region_np,(2,1,0))

def save_tumor2jpg_rotated_rect(num, ct_path, mask_path, plot=False):
    """旋转矩形进行肿瘤区域裁剪

    Args:
        num (_type_): ct_name,也就是编号,举例1-05778746-Pre.png
        ct_path (_type_): ct源文件的路径
        mask_path (_type_): mask文件的路径
        plot (bool, optional): 决定是否绘制图像. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        transform之后的图像: （3，224，224）
    """
    
    
    # 加载图像
    data = load_nii(ct_path, mask_path, num, resample=False)
    image, mask = data['image'][data['lagc_indices']], data['mask'][data['lagc_indices']]
    
    # 找到最大帧
    max_idx = np.argmax(np.sum(mask, axis=(1, 2)))
    img, msk = image[max_idx], mask[max_idx]
    
    # 找到边框
    contours, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No valid contours found")
    
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    center, size, angle = rect
    box = np.int0(cv2.boxPoints((center, (size[0]*1.05, size[1]*1.05), angle)))
    
    # 旋转图像，让矩形边框和图像轴对齐
    if angle < -45:
        angle += 90
        size = size[::-1]
    
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    rotated_msk = cv2.warpAffine(msk.astype(np.uint8), M, (w, h))
    rotated_box = np.int0(cv2.transform(np.array([box]), M)[0])
    
    # 剪裁矩形框的区域
    padding = 10
    x_min = max(0, rotated_box[:, 0].min() - padding)
    x_max = min(w, rotated_box[:, 0].max() + padding)
    y_min = max(0, rotated_box[:, 1].min() - padding)
    y_max = min(h, rotated_box[:, 1].max() + padding)
    cropped = rotated_img[y_min:y_max, x_min:x_max]
    
    # Normalize to uint8
    cropped_uint8 = ((cropped - cropped.min()) / 
                    (cropped.max() - cropped.min() + 1e-6) * 255).astype(np.uint8)
    
    # 转换成三通道并进行transform
    cropped_3d = np.repeat(cropped_uint8[None, ...], 3, axis=0).transpose(1, 2, 0)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    result = transform(cropped_3d).numpy().transpose(1, 2, 0)
    #可视化
    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(18, 5))
        
        # Original with overlay
        rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        rgb[msk > 0] = [255, 0, 0]
        cv2.polylines(rgb, [box], True, (0, 255, 0), 2)
        axs[0, 0].imshow(rgb)
        axs[0, 0].set_title("Original with Overlay")
        
        # Rotated with overlay
        rgb_rot = cv2.cvtColor(rotated_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        rgb_rot[rotated_msk > 0] = [255, 0, 0]
        cv2.polylines(rgb_rot, [rotated_box], True, (0, 255, 0), 2)
        axs[0, 1].imshow(rgb_rot)
        axs[0, 1].set_title("Rotated with Overlay")
        
        # Cropped and transformed
        axs[0, 2].imshow(cropped_uint8, cmap='gray')
        axs[0, 2].set_title("Cropped Region")
        axs[1, 0].imshow(result.squeeze(), cmap='gray')
        axs[1, 0].set_title("Transformed Result")
        axs[1, 1].imshow(cropped_3d.squeeze(), cmap='gray')
        axs[1, 1].set_title("3-Channel Cropped")
        
        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"process_CT/images/{num}.png") 
        plt.close()        
    return result.transpose(2, 1, 0)


def save_tumor2jpg_rotated_square(num, ct_path, mask_path, plot=False):
    """旋转矩形进行肿瘤区域裁剪,裁剪出以肿瘤为中心的正方形。

    Args:
        num (_type_): ct_name,也就是编号,举例1-05778746-Pre.png
        ct_path (_type_): ct源文件的路径
        mask_path (_type_): mask文件的路径
        plot (bool, optional): 决定是否绘制图像. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        transform之后的图像: （3，224，224）
    """
    
    
    # 加载图像
    data = load_nii(ct_path, mask_path, num, resample=False)
    image, mask = data['image'][data['lagc_indices']], data['mask'][data['lagc_indices']]
    
    # 找到最大帧
    max_idx = np.argmax(np.sum(mask, axis=(1, 2)))
    img, msk = image[max_idx], mask[max_idx]
    
    # 找到边框
    contours, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No valid contours found")
    
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    center, size, angle = rect
    
    # 旋转图像，让矩形边框和图像轴对齐
    if angle < -45:
        angle += 90
        size = size[::-1]
    
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    rotated_msk = cv2.warpAffine(msk.astype(np.uint8), M, (w, h))
    
    # box = np.int0(cv2.boxPoints((center, (size[0]*1.05, size[1]*1.05), angle)))
    box = np.int0(cv2.boxPoints(rect))
    rotated_box = np.int0(cv2.transform(np.array([box]), M)[0])
    
    rotated_center_x, rotated_center_y = tuple(map(int, cv2.transform(np.array([[center]]), M)[0][0]))

    final_side_length = int(max(size[0], size[1]) * 1.2) 
    square_x_min = int(rotated_center_x - final_side_length / 2)
    square_y_min = int(rotated_center_y - final_side_length / 2)
    square_x_max = int(rotated_center_x + final_side_length / 2)
    square_y_max = int(rotated_center_y + final_side_length / 2)
    
    cropped_temp = rotated_img[max(0, square_y_min):min(h, square_y_max),
                               max(0, square_x_min):min(w, square_x_max)]    
    pad_left = abs(min(0, square_x_min))
    pad_right = abs(min(0, w - square_x_max))
    pad_top = abs(min(0, square_y_min))
    pad_bottom = abs(min(0, h - square_y_max))
    
    
    cropped = cv2.copyMakeBorder(cropped_temp, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    if cropped.shape[0] != final_side_length or cropped.shape[1] != final_side_length:
        cropped = cv2.resize(cropped, (final_side_length, final_side_length), interpolation=cv2.cv2.INTER_CUBIC)

    
    cropped_uint8 = ((cropped - cropped.min()) / 
                    (cropped.max() - cropped.min() + 1e-6) * 255).astype(np.uint8)
    
    # 转换成三通道并进行transform
    cropped_3d = np.repeat(cropped_uint8[None, ...], 3, axis=0).transpose(1, 2, 0)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    result = transform(cropped_3d).numpy().transpose(1, 2, 0)
    #可视化
    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(18, 5))
        
        # Original with overlay
        rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        rgb[msk > 0] = [255, 0, 0]
        cv2.polylines(rgb, [box], True, (0, 255, 0), 2)
        axs[0, 0].imshow(rgb)
        axs[0, 0].set_title("Original with Overlay")
        
        # Rotated with overlay
        rgb_rot = cv2.cvtColor(rotated_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        rgb_rot[rotated_msk > 0] = [255, 0, 0]
        cv2.polylines(rgb_rot, [rotated_box], True, (0, 255, 0), 2)
        axs[0, 1].imshow(rgb_rot)
        axs[0, 1].set_title("Rotated with Overlay")
        
        # Cropped and transformed
        axs[0, 2].imshow(cropped_uint8, cmap='gray')
        axs[0, 2].set_title("Cropped Region")
        axs[1, 0].imshow(result.squeeze(), cmap='gray')
        axs[1, 0].set_title("Transformed Result")
        axs[1, 1].imshow(cropped_3d.squeeze(), cmap='gray')
        axs[1, 1].set_title("3-Channel Cropped")
        
        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"process_CT/images/{num}.png") 
        plt.close()        
    return result.transpose(2, 1, 0)