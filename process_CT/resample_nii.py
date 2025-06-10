# 使用B-spline插值对图像进行重采样(2,2,2)
# 并对结果进行保存nii.tar.gz文件，命名和原始文件相同，处理好的文件已经删除，需要的话重新跑这个文件
import SimpleITK as sitk
import numpy as np
from typing import List, Tuple, Optional
from os import listdir,makedirs
from os.path import join
from functions.data_process import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
def print_image_info(image: sitk.Image, name: str) -> None:
    """Print basic image information"""
    print(f"\n{name} Information:")
    print(f"  Size: {image.GetSize()}")
    print(f"  Spacing: {image.GetSpacing()}")

def calculate_output_size(input_image: sitk.Image, output_spacing: List[float]) -> List[int]:
    """Calculate output size based on physical dimensions"""
    input_size = input_image.GetSize()
    input_spacing = input_image.GetSpacing()
    return [int(round(s * sp_in / sp_out)) 
            for s, sp_in, sp_out in zip(input_size, input_spacing, output_spacing)]

def resample_image(input_image: sitk.Image, output_spacing: List[float], interpolator: int, default_pixel_value: float = 0.0) -> sitk.Image:
    """Resample image with dynamic output size"""
    output_size = calculate_output_size(input_image, output_spacing)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(output_size)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())
    resampler.SetOutputPixelType(input_image.GetPixelID())
    resampler.SetDefaultPixelValue(default_pixel_value)
    return resampler.Execute(input_image)

def get_labeled_frames(mask: sitk.Image, axis: int = 2) -> List[int]:
    """Return indices of frames with non-zero (labeled) pixels along specified axis"""
    mask_array = sitk.GetArrayFromImage(mask)
    if axis == 0:
        mask_array = np.transpose(mask_array, (1, 2, 0))
    elif axis == 1:
        mask_array = np.transpose(mask_array, (2, 0, 1))
    
    labeled_frames = [i for i in range(mask_array.shape[0]) if np.any(mask_array[i] > 0)]
    return labeled_frames

def process_and_resample_images(
    image_path: str,
    mask_path: str,
    image_output_path: str,
    mask_output_path: str,
    output_spacing: List[float] = [1.0, 1.0, 1.0],
    
) -> Tuple[Optional[sitk.Image], Optional[sitk.Image], List[int]]:
    """
    Process and resample medical images and mask, returning resampled images and labeled frames.
    
    Args:
        image_path: Path to input image
        mask_path: Path to input mask
        output_spacing: Desired spacing for resampling
        image_output_path: Path to save resampled image
        mask_output_path: Path to save resampled mask
    
    Returns:
        Tuple containing (resampled_image, resampled_mask, labeled_frames)
    """
    # Read images
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
    except RuntimeError as e:
        print(f"Failed to read images: {e}")
        return None, None, []

    # Convert image to float32 for better interpolation
    image_float = sitk.Cast(image, sitk.sitkFloat32)

    # Resample image with BSpline interpolation
    resampled_image = resample_image(
        image_float,
        output_spacing,
        sitk.sitkBSpline3,
        default_pixel_value=float(sitk.GetArrayViewFromImage(image).min())
    )

    # Resample mask with Nearest Neighbor
    resampled_mask = resample_image(
        mask,
        output_spacing,
        sitk.sitkNearestNeighbor,
        default_pixel_value=0.0
    )

    # Verify results and save
    if resampled_image and resampled_mask:
        sitk.WriteImage(resampled_image, image_output_path)
        sitk.WriteImage(resampled_mask, mask_output_path)
    else:
        print("Resampling failed")
        return None, None, []

    return resampled_image, resampled_mask

def process_images_multithreaded(
    path0: str = "/data/laicy/data/train_set/CT_data/original/",
    save_path0: str = "/data/laicy/data/train_set/CT_data/nii_resample/",
    max_workers: int = 16  # 线程池最大工作线程数
) -> None:
    """
    Process and resample images using multiple threads.
    
    Args:
        path0: Root directory of original images
        save_path0: Root directory for saving resampled images
        max_workers: Maximum number of threads
    """
    # path1 = listdir(path0)  # 模态缺失/模态齐全
    # path2 = listdir(path0 + path1[0])  # 免疫/化疗
    # path3 = listdir(path0 + path1[0] + "/" + path2[0])  # 动脉/静脉

    path1=['模态缺失']
    path2=['新辅助化疗']
    path3=['静脉期']
    tasks = []
    for path1_i in path1:
        for path2_i in path2:
            ids = get_ct_names(path0 + path1_i + "/" + path2_i + "/" + path3[0])
            root_dir = path1_i + "/" + path2_i + "/"
            for path3_i in path3:
                m = 'V' if path3_i == '静脉期' else 'A'  
                for id in ids:
                    save_root = f'{save_path0}{root_dir}{path3_i}'
                    makedirs(save_root, exist_ok=True)
                    image_path = f'{path0}{root_dir}{path3_i}/{id}-{m}.nii.gz'
                    mask_path = f'{path0}{root_dir}{path3_i}/{id}-{m}M.nii.gz'
                    out_image_path = f'{save_root}/{id}-{m}.nii.gz'
                    out_mask_path = f'{save_root}/{id}-{m}M.nii.gz'
                    print(out_mask_path)
                    tasks.append((image_path, mask_path, out_image_path, out_mask_path))

    total_tasks = len(tasks)
    print(f"Total tasks to process: {total_tasks}")

    # 使用线程池并行处理，并显示进度条
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_and_resample_images, *task): task[0] 
            for task in tasks
        }
        
        # 使用 tqdm 跟踪进度
        with tqdm(total=total_tasks, desc="Processing images") as pbar:
            for future in as_completed(future_to_task):
                image_path = future_to_task[future]
                try:
                    resampled_image, resampled_mask = future.result()
                    pbar.update(1)  # 每完成一个任务更新进度条
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    pbar.update(1)  # 即使出错也更新进度，避免卡住

if __name__ == "__main__":
    # 使用默认参数调用函数
    process_images_multithreaded()