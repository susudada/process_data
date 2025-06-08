from PIL import Image
Image.MAX_IMAGE_PIXELS=2e9
import cfg
import os
import matplotlib.pyplot as plt
from histolab.stain_normalizer import ReinhardStainNormalizer
import numpy as np 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging
import re
import cv2

def is_patches_need_saved(image, threshold1=0.75,threshold2=0.1):
    """
    判断背景是否过大,以及该区域是否含有病理组织
    imag:图片
    threshold1:判断白色背景占比的阈值
    threshold2:判断病理区域占比的阈值
    """

    if isinstance(image,Image.Image):
        image=np.array(image)
    
    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义浅白色和粉色的HSV范围
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([100, 10, 255])
    # 定义黑色的HSV范围
    lower_black = np.array([0, 0, 0]) 
    upper_black = np.array([180, 10, 50])
    # 定义灰色的HSV范围
    lower_gray = np.array([0, 0, 50])    
    upper_gray = np.array([180, 30, 200])

    # 定义粉色的HSV范围
    lower_pink = np.array([130, 10, 50])
    upper_pink = np.array([177, 255, 255])

    # 定义紫色范围
    lower_purple = np.array([270, 50, 50])  
    upper_purple = np.array([330, 255, 255])  

    # 无用部分的掩膜
    black_mask= cv2.inRange(hsv_image, lower_black, upper_black)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)

    # 有用部分的掩膜
    pink_mask = cv2.inRange(hsv_image, lower_pink, upper_pink)
    purple_mask=cv2.inRange(hsv_image, lower_purple, upper_purple)
    
    # 计算各颜色像素个数
    white_pixels = np.sum(white_mask > 0)
    black_pixels = np.sum(black_mask > 0)
    gray_pixels = np.sum(gray_mask>0)

    pink_pixels=np.sum(pink_mask>0)
    purple_pixels=np.sum(purple_mask>0)


    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
    background_ratio = (white_pixels)  / total_pixels

    # 计算有用部分的比例
    wsi_ratio=(pink_pixels+purple_pixels)/total_pixels
    # 判断是否属于背景占比过多，或者扫描到黑色边缘
    m1=background_ratio > threshold1 or black_pixels>200
    # 判断是否含有wsi区域
    m2=wsi_ratio>threshold2
    return (not m1) and m2

def is_blurry(image,threshold=30):
    """
    判断图像是否模糊
    image_path:图像,PIL
    threshold:清晰度阈值，默认100

    """
    if isinstance(image,Image.Image):
        image=np.array(image)

    if isinstance(image,np.ndarray):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        laplacian=cv2.Laplacian(gray,cv2.CV_64F)
        variance=laplacian.var()
        # print(variance)
        return variance>threshold
    else:
        print("the shape of image are not ndarray or PIL")

from PIL import Image
Image.MAX_IMAGE_PIXELS=2e9
import cfg
import os
import matplotlib.pyplot as plt
from histolab.stain_normalizer import ReinhardStainNormalizer
import numpy as np 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging
from os.path import join,exists
def seperate_to_slice_image(args,num,regions):

    # 定义normalizer
    target_path="process_wsi/tcga_luadlusc/1_0_515.png"
    target_image=Image.open(target_path).convert('RGB')
    normalizer=ReinhardStainNormalizer()
    normalizer.fit(target_image)
    #打开sub_wsi_dir
    for i,image in enumerate(regions):
        saved_count=0
        width=image.size[0]
        height=image.size[1]
        slide_dir=os.path.join(args.output_dir,"slice_image",num,str(i))
        normalized_dir=os.path.join(args.output_dir,"normalized_slice",num,str(i))

        if not os.path.exists(slide_dir):
            os.makedirs(slide_dir)
        
        if not os.path.exists(normalized_dir):
            os.makedirs(normalized_dir)
            
        for indx,x in enumerate(range(0,width,args.slice_size)):
            for indy,y in enumerate(range(0,height,args.slice_size)):
                if exists(join(slide_dir,f'{i}_{indx}_{indy}.png')) and exists(join(normalized_dir,f'{i}_{indx}_{indy}_norm.png')):
                    continue

                right=min(x+args.slice_size,width)
                lower=min(y+args.slice_size,height)

                slice_image=image.crop((x,y,right,lower)).convert('RGB')
                if slice_image.size == (args.slice_size,args.slice_size) and is_patches_need_saved(slice_image) :
                    saved_count+=1
                    if os.path.exists(os.path.join(slide_dir,f'{i}_{indx}_{indy}.png')) and os.path.exists(os.path.join(normalized_dir,f'{i}_{indx}_{indy}_norm.png')):
                        continue
                    slice_image.save(os.path.join(slide_dir,f'{i}_{indx}_{indy}.png'))
                    normalized_img=normalizer.transform(slice_image)
                    normalized_img.save(os.path.join(normalized_dir,f'{i}_{indx}_{indy}_norm.png'))
    logging.warning(f"数据名字'{num},has{saved_count}slice_image")
    return saved_count
