import os
import re


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