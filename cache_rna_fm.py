import os
import torch
import pandas as pd
from tqdm import tqdm
import fm  # <--- 直接使用您的 fm 包
from rna_backbone_design.data import utils as du

# 配置
CSV_PATH = "metadata/rna_metadata.csv"  # 请根据实际路径修改
CACHE_DIR = "data/rna_fm_embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_seq_from_aatype(aatype):
    """将 aatype 索引转换为字符序列"""
    # 假设映射: 0:A, 1:C, 2:G, 3:U (请核对您的 nucleotide_constants.py)
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
    # 过滤掉非标准碱基 (如 padding 或特殊 token)
    return "".join([mapping.get(int(x), 'N') for x in aatype])

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. 加载模型 (不使用 rhofold)
    print(f"Loading RNA-FM model on {DEVICE}...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.to(DEVICE)
    model.eval()

    # 2. 读取数据
    if not os.path.exists(CSV_PATH):
        print(f"Error: Metadata file not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # 3. 提取特征
    print("Starting extraction...")
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            pdb_name = row['pdb_name']
            save_path = os.path.join(CACHE_DIR, f"{pdb_name}.pt")
            
            # 如果已存在则跳过
            if os.path.exists(save_path): continue

            # 读取预处理好的数据以获取 aatype
            # 假设 processed_path 列存在且指向 .pkl 文件
            try:
                pkl_path = row['processed_path']
                pkl_data = du.read_pkl(pkl_path)
                aatype = pkl_data['aatype'] 
            except Exception as e:
                print(f"Skipping {pdb_name}: {e}")
                continue

            # 准备输入
            seq_str = get_seq_from_aatype(aatype)
            data = [(pdb_name, seq_str)]
            
            # 运行 RNA-FM
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(DEVICE)
            
            # 提取第 12 层特征
            results = model(batch_tokens, repr_layers=[12])
            token_embeddings = results["representations"][12] # [1, L+2, 640]
            
            # 裁剪掉 <cls> 和 <eos>
            seq_emb = token_embeddings[0, 1:-1, :].cpu() # [L, 640]
            
            # 保存
            torch.save(seq_emb, save_path)

    print("Done! Embeddings cached.")

if __name__ == "__main__":
    main()