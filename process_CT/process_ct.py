# 处理CT
# 选择肿瘤区域最大的一帧
# 包括了重采样,标准化,reshape(224,224)，只选取肿瘤区域（其他区域进行padding）
# 将每个病理中动脉期(indx=0)和静脉期(indx=1)的CT 拼接保存在npy文件中(2,3,224,224)
# 保存数据地址/home/laicy/data/train_set/CT_data/processed_ct/ 已经删除，需要的话可以重新运行
import os
import re
from functions.data_process import *
from functions.utils import *
from tqdm import tqdm
def get_ct_names(ct_path):
    #获取ct的名字，返回的是CT名的列表
    #ct_path：ct名字的列表
    #ct_names:返回的是ct名字的列表

    nums=os.listdir(ct_path)
    ct_names=[]
    for num in nums:
        match = re.match(r"(\d+-\d+-Pre)",num)
        if match:
            ct_names.append(match.group(1))
    return list(set(ct_names))

path1="/home/laicy/data/exterenal_validation_set/CT_data/original/"
path2='模态缺失'  #模态齐全or模态缺失
path3=os.listdir(os.path.join(path1,path2)) #免疫 or 化疗

for i in range(len(path3)):
    path4=os.listdir(os.path.join(path1,path2,path3[i])) #动脉 or 静脉
    ct_path=os.path.join(path1,path2,path3[i],path4[0])
    ct_names=get_ct_names(ct_path)
    output_path="/home/laicy/data/exterenal_validation_set/CT_data/processed/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for ct_name in tqdm(ct_names):
        out=[]
        outname=ct_name.replace('-Pre','')
        if os.path.exists(join(output_path,f'{outname}.npy')):
            continue
        for j in range(len(path4)):
            path=os.path.join(path1,path2,path3[i],path4[j])
            plot_path=os.path.join(path1,'pred_gif',path3[i],path4[j])
            if j==0:
                image_path=os.path.join(path,ct_name+'-A.nii.gz')
                mask_path=os.path.join(path,ct_name+'-AM.nii.gz')
                
            if j==1:
                image_path=os.path.join(path,ct_name+'-V.nii.gz')
                mask_path=os.path.join(path,ct_name+'-VM.nii.gz')
            a=save_tumor2jpg_rotated_rect(ct_name,image_path,mask_path,plot=True)    
            out.append(a)
        out=np.stack(out,axis=0).squeeze()
        assert out.shape==(2,3,224,224),f'error,the shape of out is {out.shape}'
        
        np.save(join(output_path,f'{outname}.npy'),out)