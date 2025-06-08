# 处理图片,也就是3D ResNet以及3D swin transformer需要的数据，已经删除，需要的话重新运行这个文件
# 返回的是npz,artery:[50,50,50],vein:[50,50,50],titan_features:[768,],label,flag:1模态齐全，0，模态缺失
# 保存在/home/laicy/data/train_set/CT_data/processed_3d/，已经删除，需要的话重新运行这个文件
from fmcib.visualization import visualize_seed_point
from fmcib.preprocessing import preprocess
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
import os
import re
from tqdm import tqdm

import numpy as np
from multiprocessing import Pool
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from scipy.ndimage import rotate, zoom
from functions.utils import *
from functions.plt_ct import *


def get_centroid(mask_path, cache={}):
    if mask_path not in cache:
        mask = sitk.ReadImage(mask_path)
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask)
        cache[mask_path] = label_shape_filter
    filter = cache[mask_path]
    try:
        return filter.GetCentroid(255)
    except:
        return filter.GetCentroid(1)
    
def get_ct_data(img_path,mask_path,sample_name):
    """"
    输入img_path和mask_path，返回处理后的数据[50,50,50]
    """
    data=np.zeros((50, 50, 50))        
    centroid = get_centroid(mask_path)
    data = {
        "image_path": [img_path],
        "PatientID": [sample_name],
        "coordX": centroid[0],
        "coordY": centroid[1],
        "coordZ": centroid[2],
    }
    data = preprocess(data).numpy()
    return data

def center_crop_or_pad(volume, target_shape=(50, 50, 50)):
    """将volume裁剪或填充为target_shape（中心对齐）"""
    output = np.zeros(target_shape, dtype=volume.dtype)
    in_shape = volume.shape
    offsets = [(in_shape[i] - target_shape[i]) // 2 for i in range(3)]
    for i in range(3):
        if in_shape[i] < target_shape[i]:
            # 需要填充
            pad_before = (target_shape[i] - in_shape[i]) // 2
            pad_after = target_shape[i] - in_shape[i] - pad_before
            volume = np.pad(volume, 
                            pad_width=[(pad_before, pad_after) if dim==i else (0,0) for dim in range(3)], 
                            mode='constant')
        elif in_shape[i] > target_shape[i]:
            # 需要裁剪
            start = (in_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            volume = volume[start:end, :, :] if i == 0 else \
                     volume[:, start:end, :] if i == 1 else \
                     volume[:, :, start:end]
    return volume

def augment_ct_volume(volume):
    """
    """
    if random.random() < 0.5:
        axes = random.choice([(0, 1), (0, 2), (1, 2)])
        angle = random.choice([90, 180, 270])
        volume = rotate(volume, angle, axes=axes, reshape=False, order=1)

    for axis in [0, 1, 2]:
        if random.random() < 0.3:
            volume = np.flip(volume, axis=axis)

    if random.random() < 0.3:
        zoom_factor = random.uniform(0.9, 1.1)
        volume = zoom(volume, zoom=zoom_factor, order=1)

    # 无论如何，最后都中心裁剪或填充到目标大小
    volume = center_crop_or_pad(volume, target_shape=(50, 50, 50))

    return volume

def augment_titan_features_random(features):
    """
    """
    # Step 1: 如果是 [N, 768]，随机选择一行
    if features.ndim == 2 and features.shape[1] == 768:
        idx = np.random.randint(0, features.shape[0])
        features = features[idx]
    elif features.ndim == 3 and features.shape[1] == 768:
        idx = np.random.randint(0, features.shape[0])
        features = features[idx]
    elif features.ndim == 1 and features.shape[0] == 768:
        pass
    else:
        raise ValueError(f"Unsupported titan feature shape: {features.shape}")

    # Step 2: 增强（特征 shape 现在一定是 [768,]）
    method = random.choice(['random_zero', 'select_one'])

    if method == 'random_zero':
        mask = np.random.rand(768) > 0.2  # 保留 80%
        return features * mask

    elif method == 'select_one':
        idx = np.random.randint(0, 768)
        new_feat = np.zeros_like(features)
        new_feat[idx] = features[idx]
        return new_feat

    else:
        return features

def process_single_npz2(item, output_dir, dataset_type, label_name="TRG",augment=False):
    """
    处理单个样本为NPZ格式的函数，加入label和titan_features版本
    参数:
        item: 单个样本的字典数据
        output_dir: 输出目录
        phase: 处理阶段 ('artery' 或 'vein')
        dataset_type: 数据集类型 ('Training' 或 'Test')
        plot: 是否生成可视化GIF
        label_name: 标签字段名称 (默认为 "TRG")
    
    返回:
        sample_name: 处理的样本名称
        success: 是否成功处理
    """
    sample_name = item['sample_name']
    phases=['artery','vein']
    try:
        # 根据阶段选择路径
        for phase in phases:
            if phase == 'artery':
                image_path = item['artery_path']
                mask_path = item['artery_mask']
                artery=get_ct_data(image_path,mask_path,sample_name).squeeze()
            else:
                image_path = item['vein_path']
                mask_path = item['vein_mask']
                vein=get_ct_data(image_path,mask_path,sample_name).squeeze()
        if augment:
            artery = augment_ct_volume(artery)
            vein = augment_ct_volume(vein)                
    
        titan_path=item['titan_path']
        label = item[label_name]
        if titan_path:
            titan_features=np.load(titan_path).squeeze(axis=1)
            
            if augment:
                titan_features = augment_titan_features_random(titan_features)
            else:
                titan_features=np.mean(titan_features,axis=0).squeeze()  #[768,]
        else:
            titan_features = np.zeros((768,), dtype=np.float32)
        
        assert titan_features.shape == (768,),f"error,titan_features shape:{titan_features.shape}"
        
        npz_dir = os.path.join(output_dir, dataset_type, 'npz')
                
        os.makedirs(npz_dir, exist_ok=True)
        titan_complete = 1 if titan_path else 0


        if augment:
            npz_path = os.path.join(npz_dir, f'{sample_name}_aug.npz')
        else:
            npz_path = os.path.join(npz_dir, f'{sample_name}.npz')
            
        assert artery.shape == vein.shape == (50, 50, 50), f"Shape mismatch: artery {artery.shape}, vein {vein.shape}"

        np.savez(npz_path, artery=artery, vein=vein, label=label,titan_features=titan_features,flag=titan_complete)
        
        return sample_name, True
    
    except Exception as e:
        print(f"处理样本 {sample_name} 时出错: {str(e)}")
        return sample_name, False
def process_ct_npz(json_path, output_dir, dataset_type='Training', plot=False,label_name="TRG",augment=False, max_workers=4):
    """
    多线程处理CT数据并保存为适配NPZRawDataset的.npz格式
    
    参数:
        json_path: JSON文件路径，包含样本元数据
        output_dir: 输出目录
        phase: 处理阶段 ('artery' 或 'vein')
        dataset_type: 数据集类型 ('Training' 或 'Test')
        plot: 是否生成可视化GIF
        max_workers: 最大线程数
    """
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 创建输出目录
    npz_dir = os.path.join(output_dir, dataset_type, 'npz')
    plot_dir = os.path.join(output_dir, dataset_type, 'plots') if plot else None
    
    os.makedirs(npz_dir, exist_ok=True)
    if plot and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    
    # 使用ThreadPoolExecutor进行多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in data:
            futures.append(executor.submit(
                process_single_npz2,
                item, output_dir,dataset_type,label_name=label_name,augment=augment
            ))
        
        # 使用tqdm显示进度
        success_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {dataset_type} samples"):
            sample_name, success = future.result()
            if success:
                success_count += 1
        
        print(f"Finished processing {dataset_type} phase. Success: {success_count}/{len(data)}")

if __name__ == "__main__":
    json_path = "/home/laicy/data/train_set/CT_data/data_splits/response/all_dataset_ex.json"
    output_dir = "/home/laicy/data/train_set/CT_data/processed_3d/response/"
    
    max_workers = 16
    
    # 处理数据
    process_ct_npz(json_path, output_dir, dataset_type='Val_all_ex', plot=False,label_name="Response（0 no response,1 response）",augment=False, max_workers=max_workers)