import argparse
import os

def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default="/home/laicy/data/train_set/wsi_data/新辅助化疗/", help='data_path of wsi svs')
    parser.add_argument('-output_dir',type=str,default="process_wsi",help='output path of WSI data_process')
    parser.add_argument('-nums', type=list, default=[], help='list of filenames in data_dir')
    parser.add_argument('-level',type=int,default=1,help='level of svs')
    parser.add_argument('-slice_size',type=int,default=1024,help='the size of reading wsi_slice')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads to use (default: 4)')
    opt = parser.parse_args()

    # if not opt.nums:
    #     opt.nums=os.listdir(opt.output_dir)
    if not opt.nums:
        opt.nums=[os.path.splitext(p)[0]  for p in os.listdir(opt.data_dir)
                  if os.path.splitext(p)[-1] in ['.svs']]
    return opt
