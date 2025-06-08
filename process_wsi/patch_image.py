# 根据分割slice图片进行分割
import os
import re
from PIL import Image
import cfg
import cv2
import numpy as np
def is_image_corrupted(file_path):
    """
    使用 OpenCV 检查图片是否损坏
    :param file_path: 图片文件路径
    :return: True（损坏）或 False（未损坏）
    """
    try:
        # 尝试读取图片
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read image: {file_path}")
            return True  # 如果读取失败，图片损坏
        return False  # 如果读取成功，图片未损坏
    except Exception as e:
        print(f"Error reading image: {file_path} - {e}")
        return True  # 如果抛出异常，图片损坏
# 提取图片文件名中的坐标
def extract_coordinates(filename):
    match = re.match(r"^0_(\d+)_(\d+)_norm\.png", filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    return None,None

# 统一调整图片尺寸
def resize_image(img, size=(224, 224)):
    """
    调整图片尺寸
    :param img: OpenCV 图像对象
    :param size: 目标尺寸 (width, height)
    :return: 调整后的图像
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

# 拼接图片函数
def stitch_images(data_dir, num):
    """
    拼接图片
    :param data_dir: 数据目录
    :param num: 编号
    """
    image_folder = os.path.join("process_wsi/normalized_slice", num, str(0))
    files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    # 提取图片的坐标并排序
    coordinates = [extract_coordinates(f) for f in files]
    coordinates = [coord for coord in coordinates if coord != (None, None)]
    
    if not coordinates:
        print("没有找到符合命名规则的图片。")
        return
    
    # 假设每张图片调整后的大小是 224x224
    img_width, img_height = 224, 224
    
    # 计算拼接后图像的大小
    max_x = max(coord[0] for coord in coordinates)
    max_y = max(coord[1] for coord in coordinates)
    
    # 创建一个新的空白大图
    stitched_image = np.zeros(((max_y + 1) * img_height, (max_x + 1) * img_width, 3), dtype=np.uint8)
    
    # 加载并处理每张图片，调整尺寸并粘贴到正确的位置
    for filename in files:
        x, y = extract_coordinates(filename)
        if x is not None and y is not None:
            file_path = os.path.join(image_folder, filename)
            
            # 检查图片是否损坏
            if is_image_corrupted(file_path):
                print(f"Skipping corrupted image: {filename}")
                continue  # 跳过损坏的图片
            
            try:
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Failed to load image: {filename}")
                    continue  # 如果加载失败，跳过该图片
                
                img = resize_image(img, size=(img_width, img_height))  # 调整图片尺寸
                stitched_image[y * img_height:(y + 1) * img_height,
                              x * img_width:(x + 1) * img_width] = img
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # 保存拼接后的图像
    output_path = os.path.join("process_wsi", f"{num}.png")
    cv2.imwrite(output_path, stitched_image)
    print(f"Stitched image saved to: {output_path}")

    # 显示拼接后的图像
# 调用函数
args=cfg.parse_args()
# for num in args.nums:
for num in ['1-12461215-H-A1']:
    stitch_images(args.output_dir,num)
