# 读取excel文件，划分成训练集和测试集，每一条信息中包含了数据两个期相的地址（data&mask）
# 可以指label列，根据不同的label进行划分，默认是TRG，也就是column=1
import os
from os.path import join
import re
import random
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold
random.seed(42)

# .nii所在的大文件夹
BASE_DIR="/home/laicy/data/exterenal_validation_set/CT_data/nii_resample/"  #修改
OUTPUT_DIR = "/home/laicy/data/train_set/CT_data/data_splits/response"
TITAN_DIR="/home/laicy/data/exterenal_validation_set/titan_features/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def parse_excel_info(filepath):
    """解析Excel文件名获取治疗方式和模态类型"""
    filename = os.path.basename(filepath)
    parts = filename.split()
    return {
        'treatment': parts[0],  # 新辅助免疫/新辅助化疗
        'modality': parts[1],   # 模态缺失/模态齐全
        'excel_path': filename
    }

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

def load_and_clean_excels(excel_files):
    """加载并清洗Excel数据"""
    dfs = []
    
    for filepath in excel_files:
        info = parse_excel_info(filepath)
        try:
            df = pd.read_excel(filepath,dtype={0:str,1:str})
            
            # 生成样本名称，并确保其为字符串类型
            sample_name_1 = df.iloc[:, 0].astype(str).str.strip()
            sample_name_2 = df.iloc[:, 1].astype(str).str.strip()
            df['sample_name'] = sample_name_1.values + "-" + sample_name_2.values 
            
            # 添加元数据
            df['treatment'] = info['treatment']
            df['modality'] = info['modality']
            
            # 处理指标列（从第4列开始）
            metric_cols = df.columns[3:]
            for col in metric_cols:
                # 统一空值表示
                df[col] = df[col].replace(['', 'NA', 'NaN', 'None'], np.nan)
                # 尝试转换为数值类型
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass
            
            # 保留需要的列
            keep_cols = metric_cols.tolist()
            dfs.append(df[keep_cols])
            
        except Exception as e:
            print(f"错误: 加载 {filepath} 失败 - {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("所有Excel文件加载失败")
    
    return pd.concat(dfs, ignore_index=True)

def build_file_paths(base_name, modality, treatment):
    """构建文件路径字典"""
    base_dir = os.path.join(BASE_DIR, modality, treatment)
    # 动脉期文件
    artery_file = f"{base_name}-Pre-A.nii.gz"
    artery_mask = f"{base_name}-Pre-AM.nii.gz"

    # 静脉期文件
    vein_file = f"{base_name}-Pre-V.nii.gz"
    vein_mask = f"{base_name}-Pre-VM.nii.gz"
    titan_path=os.path.join(TITAN_DIR, f"{base_name}.npy")
    if os.path.exists(titan_path):
        titan_path=titan_path
    else:
        titan_path=None                    
    return {
        'artery_path': os.path.join(base_dir, "动脉期", artery_file),
        'artery_mask': os.path.join(base_dir, "动脉期", artery_mask),
        'vein_path': os.path.join(base_dir, "静脉期", vein_file),
        'vein_mask': os.path.join(base_dir, "静脉期", vein_mask),
        "titan_path":titan_path
    }

def verify_files(file_dict):
    """验证文件是否存在"""
    return all(os.path.exists(path) for path in file_dict.values())

def create_dataset(df):
    """创建最终数据集,并处理样本缺失的情况"""
    records = []
    missing_files = []
    
    for _, row in df.iterrows():
        try:
            # 构建文件路径
            paths = build_file_paths(row['sample_name'], row['modality'], row['treatment'])
            # 检查文件是否存在
            missing = [k for k, v in list(paths.items())[:-1] if not os.path.exists(v)]
            if missing:
                missing_files.append({
                    'sample': row['sample_name'],
                    'missing_files': missing,
                    'treatment': row['treatment'],
                    'modality': row['modality']
                })
                continue
            
            # 构建记录
            record = {
                'sample_name': row['sample_name'],
                'treatment': row['treatment'],
                'modality': row['modality'],
                **paths
            }
            
            # 添加指标数据（自动处理NaN）
            for col in df.columns:
                if col not in ['sample_name', 'treatment', 'modality']:
                    record[col] = row[col] if pd.notna(row[col]) else None
            
            records.append(record)
            
        except Exception as e:
            print(f"处理样本 {row.get('sample_name', '未知')} 出错: {str(e)}")
            continue
    
    # 保存缺失文件信息
    if missing_files:
        missing_report = os.path.join(OUTPUT_DIR, "missing_files_report.json")
        with open(missing_report, 'w') as f:
            json.dump(missing_files[:100], f, indent=4,ensure_ascii=False)  # 最多保存100条记录
        print(f"\n警告: 共 {len(missing_files)} 个样本缺失文件，详见 {missing_report}")
    
    return pd.DataFrame(records), missing_files

def clean_dataframe(df, metric_col, drop_na=True):
    """
    清洗DataFrame，处理缺失值并检查列索引合法性

    参数:
    - df: pandas DataFrame
    - metric_col: 单个列名/索引，或列名/索引的列表
    - drop_na: 是否删除缺失值行（默认True）

    返回:
    - 清洗后的DataFrame
    """
    import pandas as pd

    # 标准化 metric_col 为列表
    if isinstance(metric_col, (int, str)):
        metric_cols = [metric_col]
    elif isinstance(metric_col, list):
        metric_cols = metric_col
    else:
        raise TypeError("metric_col 应为 int、str 或其列表")

    # 检查列索引合法性
    valid_cols = df.columns
    for col in metric_cols:
        if isinstance(col, int):
            if col >= len(valid_cols):
                raise ValueError(f"metric_col 索引 {col} 超出范围 (最大索引: {len(valid_cols)-1})")
        elif isinstance(col, str):
            if col not in valid_cols:
                raise ValueError(f"metric_col 名称 '{col}' 不存在于 DataFrame 列中")
        else:
            raise TypeError(f"metric_col 中的元素必须为 int 或 str，但收到: {type(col)}")

    # 获取列名（无论 metric_col 是索引还是列名）
    metric_col_names = [
        df.columns[col] if isinstance(col, int) else col
        for col in metric_cols
    ]

    if drop_na:
        df_clean = df.dropna(subset=metric_col_names)
        dropped = len(df) - len(df_clean)
        if dropped > 0:
            print(f"⚠️ 删除了 {dropped} 行包含NaN值的数据")
    else:
        if df[metric_col_names].isna().any().any():
            raise ValueError("指定的指标列中包含 NaN，请设置 drop_na=True 或手动处理")
        df_clean = df.copy()

    return df_clean

def stratified_split_by_metrics(df, metric_col=[8,-1], test_size=0.2, drop_na=True):
    """
    基于指定指标列划分训练集和测试集，并统计不同类指标的个数
    """
    RANDOM_SEED = 42
    print("📋 原始数据预览:")
    print(df.head())

    df_clean = clean_dataframe(df, metric_col, drop_na)
    strata = df_clean.iloc[:, metric_col].astype(str)

    print("\n📊 原始数据中各类指标的个数:")
    print(strata.value_counts())

    train_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=strata
    )

    print("\n✅ 训练集中各类指标的个数:")
    print(train_df.iloc[:, metric_col].astype(str).value_counts())
    print("\n✅ 测试集中各类指标的个数:")
    print(test_df.iloc[:, metric_col].astype(str).value_counts())

    return train_df, test_df
def kfold_split(df, n_splits=5, stat_col=8, drop_na=True):
    """
    第7列是titan_path（模态地址），第8列是TRG标签。
    划分每折包含模态齐全子集和平衡的整体数据集。
    
    参数:
    - df: 输入DataFrame
    - n_splits: 折数
    - stat_col: 用于分层的列索引或列名
    - drop_na: 是否删除包含NaN的行
    
    返回:
    - 包含各折数据的列表，每折包含train_full, val_full, train_all, val_all
    """
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    
    df_clean = clean_dataframe(df, stat_col, drop_na)
    stat_col_name = df_clean.columns[stat_col] if isinstance(stat_col, int) else stat_col
    titan_col_name = df_clean.columns[7] if isinstance(7, int) else 7  # 第7列是titan_path

    # 拆分齐全/缺失模态数据
    full_modal_df = df_clean[df_clean[titan_col_name].notna()].copy()

    missing_modal_df = df_clean.copy()

    # 检查模态齐全数据的类别分布
    strata = full_modal_df[stat_col_name]
    min_class_size = strata.value_counts().min()
    if min_class_size < n_splits:
        raise ValueError(f"❌ 模态齐全数据中，最小类别样本数为 {min_class_size}，小于 n_splits={n_splits}，请降低折数或检查数据。")

    # 先对模态齐全数据进行分层划分
    full_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    full_folds = []
    for _, val_idx in full_kf.split(full_modal_df, strata):
        full_folds.append(val_idx)

    # 然后对整个数据集进行分层划分
    all_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_folds = []
    for _, val_idx in all_kf.split(missing_modal_df, missing_modal_df[stat_col_name]):
        all_folds.append(val_idx)

    folds = []
    print("\n📁 开始 K 折划分:")
    
    for i in range(n_splits):
        # 模态齐全数据的划分
        val_full_idx = full_folds[i]
        train_full_idx = [idx for fold in full_folds[:i] + full_folds[i+1:] for idx in fold]
        
        val_full = full_modal_df.iloc[val_full_idx].reset_index(drop=True)
        train_full = full_modal_df.iloc[train_full_idx].reset_index(drop=True)
        
        # 整个数据集（含缺失模态）的划分
        val_all_idx = all_folds[i]
        train_all_idx = [idx for fold in all_folds[:i] + all_folds[i+1:] for idx in fold]
        
        val_all = missing_modal_df.iloc[val_all_idx].reset_index(drop=True)
        train_all = missing_modal_df.iloc[train_all_idx].reset_index(drop=True)
        
        # 确保齐全数据是完整数据的子集
        # 通过索引映射来保证这个关系
        full_in_all_train = train_full.index.to_series().map(lambda x: x in train_all.index).all()
        full_in_all_val = val_full.index.to_series().map(lambda x: x in val_all.index).all()
        
        if not (full_in_all_train and full_in_all_val):
            # 如果不满足子集关系，调整完整数据集划分
            train_all = pd.concat([train_all, train_full]).drop_duplicates().reset_index(drop=True)
            val_all = pd.concat([val_all, val_full]).drop_duplicates().reset_index(drop=True)
            # 重新进行分层划分以保持类别平衡
            train_all_idx, val_all_idx = next(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42+i).split(missing_modal_df, missing_modal_df[stat_col_name]))
            train_all = missing_modal_df.iloc[train_all_idx].reset_index(drop=True)
            val_all = missing_modal_df.iloc[val_all_idx].reset_index(drop=True)
            # 再次确保子集关系
            train_all = pd.concat([train_all, train_full]).drop_duplicates().reset_index(drop=True)
            val_all = pd.concat([val_all, val_full]).drop_duplicates().reset_index(drop=True)

        print(f"\n📂 Fold {i+1}:")
        print(f"模态齐全训练集大小: {len(train_full)}, 类别分布:")
        print(train_full[stat_col_name].value_counts().sort_index())
        print(f"模态齐全验证集大小: {len(val_full)}, 类别分布:")
        print(val_full[stat_col_name].value_counts().sort_index())
        
        print(f"模态含缺失训练集大小: {len(train_all)}, 类别分布:")
        print(train_all[stat_col_name].value_counts().sort_index())
        print(f"模态含缺失验证集大小: {len(val_all)}, 类别分布:")
        print(val_all[stat_col_name].value_counts().sort_index())

        folds.append({
            "fold_id": i + 1,
            "train_full": train_full,
            "val_full": val_full,
            "train_all": train_all,
            "val_all": val_all
        })

    return folds

def get_complete_data(df, stat_col=8, drop_na=True):
    """
    返回给定指标都存在的所有数据，以及titan_path存在的所有数据
    
    参数:
    - df: 输入DataFrame
    - stat_col: 用于检查的列索引或列名
    - drop_na: 是否删除包含NaN的行
    
    返回:
    - 包含完整数据和模态齐全数据的字典
    """
    import pandas as pd
    
    df_clean = clean_dataframe(df, stat_col, drop_na)

    stat_col_name = df_clean.columns[stat_col] if isinstance(stat_col, int) else stat_col
    titan_col_name = df_clean.columns[7] if isinstance(7, int) else 7  # 第7列是titan_path

    # 获取所有给定指标都存在的数据（完整数据）
    complete_df = df_clean.copy()
    
    # 获取titan_path存在的数据（模态齐全数据）
    full_modal_df = df_clean[df_clean[titan_col_name].notna()].copy()

    print("\n📊 数据统计信息:")
    print(f"完整数据集大小: {len(complete_df)}, 类别分布:")
    print(complete_df[stat_col_name].value_counts().sort_index())
    print(f"模态齐全数据集大小: {len(full_modal_df)}, 类别分布:")
    print(full_modal_df[stat_col_name].value_counts().sort_index())

    return {
        "complete_data": complete_df,
        "full_modal_data": full_modal_df
    }

if __name__ == '__main__':

    excel_files={
        "/home/laicy/data/exterenal_validation_set/CT_data/新辅助化疗 模态齐全 预测终点.xlsx",
        "/home/laicy/data/exterenal_validation_set/CT_data/新辅助化疗 模态缺失 预测终点.xlsx",
        "/home/laicy/data/exterenal_validation_set/CT_data/新辅助免疫 模态齐全 预测终点.xlsx",
        "/home/laicy/data/exterenal_validation_set/CT_data/新辅助免疫 模态缺失 预测终点.xlsx",
    }
    combined_df=load_and_clean_excels(excel_files)
    print(f"from {len(excel_files)} excel files, {len(combined_df)} records loaded.")
    full_dataset,missing=create_dataset(combined_df)
    if len(full_dataset)==0:
        raise ValueError("No data found after combining all excel files.")

    # folds = kfold_split(full_dataset,stat_col=-1)
    
    
    # # 保存交叉验证的拆分结果
    # for i, fold in enumerate(folds):
    #     fold["train_full"].to_json(join(OUTPUT_DIR,f"fold_{i+1}_train_full.json"), orient='records', indent=4,force_ascii=False)
    #     fold["val_full"].to_json(join(OUTPUT_DIR,f"fold_{i+1}_val_full.json"), orient='records', indent=4,force_ascii=False)
    #     fold["train_all"].to_json(join(OUTPUT_DIR,f"fold_{i+1}_train_all.json"), orient='records', indent=4,force_ascii=False)
    #     fold["val_all"].to_json(join(OUTPUT_DIR,f"fold_{i+1}_val_all.json"), orient='records', indent=4,force_ascii=False)
    #     fold_num = 1

    # 外部验证集时 取消注释
    data=get_complete_data(full_dataset)
    full_modal_data=data["full_modal_data"]
    complete_data=data["complete_data"]        
    # 外部验证集保存JSON格式
    complete_data.to_json(os.path.join(OUTPUT_DIR, "all_dataset_ex.json"), orient='records', indent=4,force_ascii=False)  #所有数据集
    full_modal_data.to_json(os.path.join(OUTPUT_DIR, "full_dataset_ex.json"), orient='records', indent=4,force_ascii=False) #模态齐全的数据