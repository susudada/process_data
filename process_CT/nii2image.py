"""
根据分配好的json文件，将CT数据处理成图像和掩码，并进行保存
-image: 输出图像目录
-annotation:输出掩码目录
-npz:将image和annotation合并成字典，保存成npz文件，并保存在npz文件下面
-phase：处理阶段（artery或vein）
-dataset_type：数据集类型（Training或Test）
-plot：是否生成可视化GIF
带com的是模态齐全的数据
"""
from functions.data_process import load_nii, mask_to_box, show_box
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join
import matplotlib.colors as mcolors
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
def process_single_sample(item, output_dir, phase, dataset_type, plot,label_name="TRG"):
    """
    处理单个样本的函数，将被多线程调用
    
    参数:
        item: 单个样本的字典数据
        output_dir: 输出目录
        phase: 处理阶段 ('artery' 或 'vein')
        dataset_type: 数据集类型 ('Training' 或 'Test')
        plot: 是否生成可视化GIF
    
    返回:
        sample_name: 处理的样本名称
        success: 是否成功处理
    """
    sample_name = item['sample_name']
    
    try:
        # 根据阶段选择路径
        if phase == 'artery':
            image_path = item['artery_path']
            mask_path = item['artery_mask']
        else:
            image_path = item['vein_path']
            mask_path = item['vein_mask']
        # 使用load_nii函数加载数据
        ct_data = load_nii(image_path, mask_path, sample_name, resample=False, plot=plot)
        image = ct_data['image']
        mask = ct_data['mask']
        lagc_indices = ct_data['lagc_indices']
        
        # 创建输出目录
        image_dir = os.path.join(output_dir, phase, dataset_type, 'image')
        mask_image_dir = os.path.join(output_dir, phase, dataset_type, 'annotation')
        plot_dir = os.path.join(output_dir, phase, dataset_type, 'plots') if plot else None
        
        sample_image_dir = os.path.join(image_dir, sample_name)
        sample_mask_jpg_dir = os.path.join(mask_image_dir, sample_name)
        
        os.makedirs(sample_image_dir, exist_ok=True)
        os.makedirs(sample_mask_jpg_dir, exist_ok=True)
        
        # 只处理有标注的帧
        for i, frame_idx in enumerate(lagc_indices):
            frame = image[frame_idx]              # 原始图像（HWC, uint8）
            frame_mask = mask[frame_idx]          # 掩码图像（HW, bool 或 0/1）

            # 保存原始图像
            frame_img = Image.fromarray(frame)
            if frame_img.mode != "RGB":
                frame_img = frame_img.convert('RGB')
            frame_img.save(os.path.join(sample_image_dir, f'{i:05d}.jpg'))

            # 创建彩色掩码图（红色表示标注区域）
            binary_mask = (frame_mask > 0).astype(np.uint8) * 255
            assert np.max(binary_mask) == 255,f"{sample_name},its max value is not 255,{np.max(binary_mask)}"
            binary_mask_img = Image.fromarray(binary_mask, mode='L')  # 显式指定为灰度模式
            binary_mask_img.save(
                os.path.join(sample_mask_jpg_dir, f'{i:05d}.png'),
                compress_level=9,
                optimize=True
            )
        
        # 如果需要生成可视化GIF
        if plot and plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            frames = []
            for i, frame_idx in enumerate(lagc_indices):
                plt.clf()
                if i == 0:
                    box = mask_to_box(mask[frame_idx])
                    if box is not None:
                        show_box(box, plt.gca())
                
                cmap = mcolors.ListedColormap(['white', 'red'])
                plt.imshow(image[frame_idx], cmap='gray')
                plt.imshow(mask[frame_idx], cmap=cmap, alpha=0.3)
                plt.savefig('tem.png', bbox_inches='tight', pad_inches=0)
                img = Image.open('tem.png')
                frames.append(img.copy())
                img.close()
            
            if frames:
                frames[0].save(
                    join(plot_dir, f'{sample_name}_truth.gif'),
                    save_all=True,
                    append_images=frames[1:],
                    duration=170,
                    loop=0
                )
        
        return sample_name, True
    
    except Exception as e:
        print(f"Error processing {sample_name}: {str(e)}")
        return sample_name, False

def process_single_sample2(item, output_dir, phase, dataset_type, plot, label_name="TRG"):
    """
    处理单个样本的函数，将被多线程调用
    
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
    
    try:
        # 根据阶段选择路径
        if phase == 'artery':
            image_path = item['artery_path']
            mask_path = item['artery_mask']
        else:
            image_path = item['vein_path']
            mask_path = item['vein_mask']
        
        # 获取标签值
        label = item[label_name]
        
        # 使用load_nii函数加载数据
        ct_data = load_nii(image_path, mask_path, sample_name, resample=False, plot=plot)
        image = ct_data['image']
        mask = ct_data['mask']
        lagc_indices = ct_data['lagc_indices']
        
        # 创建输出目录
        image_dir = os.path.join(output_dir, phase, dataset_type, 'image')
        mask_image_dir = os.path.join(output_dir, phase, dataset_type, 'annotation')
        
        sample_image_dir = os.path.join(image_dir, sample_name)
        sample_mask_jpg_dir = os.path.join(mask_image_dir, sample_name)
        
        os.makedirs(sample_image_dir, exist_ok=True)
        os.makedirs(sample_mask_jpg_dir, exist_ok=True)
        
        # 只处理有标注的帧
        for i, frame_idx in enumerate(lagc_indices):
            frame = image[frame_idx]              # 原始图像（HWC, uint8）
            frame_mask = mask[frame_idx]          # 掩码图像（HW, bool 或 0/1）

            # 保存原始图像
            frame_img = Image.fromarray(frame)
            if frame_img.mode != "RGB":
                frame_img = frame_img.convert('RGB')
            frame_img.save(os.path.join(sample_image_dir, f'{i:05d}.jpg'))

            # 将mask中非零区域替换为label值
            processed_mask = np.where(frame_mask > 0, label*10+10, 0).astype(np.uint8)
            
            # 创建彩色掩码图（使用label值表示标注区域）
            mask_img = Image.fromarray(processed_mask, mode='L')  # 显式指定为灰度模式
            mask_img.save(
                os.path.join(sample_mask_jpg_dir, f'{i:05d}.png'),
                compress_level=9,
                optimize=True
            )
            
        return sample_name, True
    
    except Exception as e:
        print(f"处理样本 {sample_name} 时出错: {str(e)}")
        return sample_name, False

def process_ct_image(json_path, output_dir, phase='artery', dataset_type='Training', plot=False, max_workers=4):
    """
    多线程处理CT数据并保存为图像和掩码-image(jpg)和-annotation(png)
    
    参数:
        json_path: JSON文件路径
        output_dir: 输出目录
        phase: 处理阶段 ('artery' 或 'vein')
        dataset_type: 数据集类型 ('Training' 或 'Test')
        plot: 是否生成可视化GIF
        max_workers: 最大线程数
    """
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 创建输出目录结构
    image_dir = os.path.join(output_dir, phase, dataset_type, 'image')
    mask_image_dir = os.path.join(output_dir, phase, dataset_type, 'annotation')
    plot_dir = os.path.join(output_dir, phase, dataset_type, 'plots') if plot else None
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_image_dir, exist_ok=True)
    if plot and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    
    # 使用ThreadPoolExecutor进行多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in data:
            futures.append(executor.submit(
                process_single_sample, 
                item, output_dir, phase, dataset_type, plot
            ))
        
        # 使用tqdm显示进度
        success_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {phase} samples"):
            sample_name, success = future.result()
            if success:
                success_count += 1
        
        print(f"Finished processing {phase} phase. Success: {success_count}/{len(data)}")

def process_single_npz(item, output_dir, phase, dataset_type, plot,label_name="TRG"):
    """
    处理单个样本为NPZ格式的函数，将被多线程调用,无label版本，与最原始的MEDSAM需要的数据结构完全一致
    
    参数:
        item: 单个样本的字典数据
        output_dir: 输出目录
        phase: 处理阶段 ('artery' 或 'vein')
        dataset_type: 数据集类型 ('Training' 或 'Test')
        plot: 是否生成可视化GIF
    
    返回:
        sample_name: 处理的样本名称
        success: 是否成功处理
    """
    sample_name = item['sample_name']
    
    try:
        # 根据阶段选择路径
        if phase == 'artery':
            image_path = item['artery_path']
            mask_path = item['artery_mask']
        else:
            image_path = item['vein_path']
            mask_path = item['vein_mask']
        
        # 加载CT数据
        label=item[label_name]
        ct_data = load_nii(image_path, mask_path, sample_name, resample=False, plot=plot)
        image = ct_data['image']  # 形状: (帧数, 高度, 宽度)
        mask = ct_data['mask']    # 形状: (帧数, 高度, 宽度)
        lagc_indices = ct_data['lagc_indices']  # 有效帧索引
        
        # 只处理有标注的帧
        valid_frames = image[lagc_indices]  # 形状: (有效帧数, 高度, 宽度)
        valid_masks = mask[lagc_indices]    # 形状: (有效帧数, 高度, 宽度)
        
        if valid_frames.dtype != np.uint8:
            valid_frames = np.clip(valid_frames, 0, np.max(valid_frames))
            valid_frames = (valid_frames / np.max(valid_frames) * 255).astype(np.uint8)

        if valid_masks.dtype != np.uint8:
            valid_masks = valid_masks.astype(np.uint8)
        
        # 创建输出目录
        npz_dir = os.path.join(output_dir, phase, dataset_type, 'npz')
        plot_dir = os.path.join(output_dir, phase, dataset_type, 'plots') if plot else None
        
        os.makedirs(npz_dir, exist_ok=True)
        
        # 保存为.npz文件
        npz_path = os.path.join(npz_dir, f'{sample_name}.npz')
        np.savez(npz_path, imgs=valid_frames, gts=valid_masks,label=label)
        
        # 如果需要生成可视化GIF
        if plot and plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            frames = []
            for i, frame_idx in enumerate(lagc_indices):
                plt.clf()
                if i == 0:
                    box = mask_to_box(mask[frame_idx])
                    if box is not None:
                        show_box(box, plt.gca())
                
                cmap = mcolors.ListedColormap(['white', 'red'])
                plt.imshow(image[frame_idx], cmap='gray')
                plt.imshow(mask[frame_idx], cmap=cmap, alpha=0.3)
                plt.savefig('tem.png', bbox_inches='tight', pad_inches=0)
                img = Image.open('tem.png')
                frames.append(img.copy())
                img.close()
            
            if frames:
                frames[0].save(
                    join(plot_dir, f'{sample_name}_truth.gif'),
                    save_all=True,
                    append_images=frames[1:],
                    duration=170,
                    loop=0
                )
        
        return sample_name, True
    
    except Exception as e:
        print(f"错误处理样本 {sample_name}: {str(e)}")
        return sample_name, False
def process_single_npz2(item, output_dir, phase, dataset_type, plot, label_name="TRG"):
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
    
    try:
        # 根据阶段选择路径
        if phase == 'artery':
            image_path = item['artery_path']
            mask_path = item['artery_mask']
        else:
            image_path = item['vein_path']
            mask_path = item['vein_mask']
        titan_path=item['titan_path']
        # 加载CT数据
        label = item[label_name]  # 获取标签值
        
        if titan_path:
            titan_features=np.load(titan_path)
            titan_features=np.mean(titan_features,axis=0).squeeze()  #[768,]
        else:
            titan_features = np.zeros((768,), dtype=np.float32)
        ct_data = load_nii(image_path, mask_path, sample_name, resample=False, plot=plot)
        image = ct_data['image']  # 形状: (帧数, 高度, 宽度)
        mask = ct_data['mask']    # 形状: (帧数, 高度, 宽度)
        lagc_indices = ct_data['lagc_indices']  # 有效帧索引
        
        # 只处理有标注的帧
        valid_frames = image[lagc_indices]  # 形状: (有效帧数, 高度, 宽度)
        valid_masks = mask[lagc_indices]    # 形状: (有效帧数, 高度, 宽度)
        
        # 修改：将mask中所有非零值替换为label值
        # valid_masks = np.where(valid_masks > 0, label+100, 0)
        
        if valid_frames.dtype != np.uint8:
            valid_frames = np.clip(valid_frames, 0, np.max(valid_frames))
            valid_frames = (valid_frames / np.max(valid_frames) * 255).astype(np.uint8)

        if valid_masks.dtype != np.uint8:
            valid_masks = valid_masks.astype(np.uint8)
        
        # 创建输出目录
        npz_dir = os.path.join(output_dir, phase, dataset_type, 'npz')
        plot_dir = os.path.join(output_dir, phase, dataset_type, 'plots') if plot else None
        
        os.makedirs(npz_dir, exist_ok=True)
        titan_complete = 1 if titan_path else 0 #修改：可以标记标记
        # 保存为.npz文件
        npz_path = os.path.join(npz_dir, f'{sample_name}.npz')
        np.savez(npz_path, imgs=valid_frames, gts=valid_masks, label=label,titan_features=titan_features,flag=titan_complete)
        
        return sample_name, True
    
    except Exception as e:
        print(f"处理样本 {sample_name} 时出错: {str(e)}")
        return sample_name, False
def process_ct_npz(json_path, output_dir, phase='artery', dataset_type='Training', plot=False,label_name="TRG", max_workers=4):
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
    npz_dir = os.path.join(output_dir, phase, dataset_type, 'npz')
    plot_dir = os.path.join(output_dir, phase, dataset_type, 'plots') if plot else None
    
    os.makedirs(npz_dir, exist_ok=True)
    if plot and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    
    # 使用ThreadPoolExecutor进行多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in data:
            futures.append(executor.submit(
                process_single_npz2,
                item, output_dir, phase, dataset_type, plot,label_name
            ))
        
        # 使用tqdm显示进度
        success_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {phase} samples"):
            sample_name, success = future.result()
            if success:
                success_count += 1
        
        print(f"Finished processing {phase} phase. Success: {success_count}/{len(data)}")

if __name__ == "__main__":
    json_path = "/home/laicy/data/train_set/CT_data/data_splits/response/full_dataset_ex.json"
    output_dir = "/home/laicy/data/train_set/CT_data/MedSam_2/response"
    
    max_workers = 16 
    
    # 处理动脉期数据              如果要保存成图片，将process_ct_npz换成process_ct_image
    process_ct_npz(json_path, output_dir, phase='artery', dataset_type='full_ex', plot=False,label_name="Response（0 no response,1 response）", max_workers=max_workers)
    # 处理静脉期数据
    process_ct_npz(json_path, output_dir, phase='vein', dataset_type='full_ex', plot=False,label_name="Response（0 no response,1 response）", max_workers=max_workers)