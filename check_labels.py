"""
快速查看标注状态
"""

import pandas as pd
import os
from config import *

def check_labels():
    """检查标注状态"""
    print("=" * 80)
    print("标注状态检查")
    print("=" * 80)
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"\n错误: 数据文件不存在 {RAW_DATA_PATH}")
        print("\n运行: python 1_data_labeling.py 生成数据")
        return
    
    df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')
    
    n_can_merge = len(df[df[TARGET] == 1])
    n_cannot_merge = len(df[df[TARGET] == 0])
    n_unlabeled = len(df[df[TARGET] == -1])
    
    print(f"\n数据总量: {len(df)}")
    print(f"\n标注进度:")
    print(f"  ✓ 可以自动合并:   {n_can_merge:4d} ({n_can_merge/len(df)*100:5.1f}%)")
    print(f"  ✗ 不可以自动合并: {n_cannot_merge:4d} ({n_cannot_merge/len(df)*100:5.1f}%)")
    print(f"  ? 未标注:         {n_unlabeled:4d} ({n_unlabeled/len(df)*100:5.1f}%)")
    
    labeled_count = n_can_merge + n_cannot_merge
    if labeled_count > 0:
        merge_ratio = n_can_merge / labeled_count * 100
        print(f"\n已标注: {labeled_count}/{len(df)} ({labeled_count/len(df)*100:.1f}%)")
        print(f"  - 自动合并比例: {merge_ratio:.1f}%")
        
        # 样本平衡检查
        if merge_ratio < 30:
            print(f"\n  ⚠️  警告: 可合并样本过少 ({merge_ratio:.1f}%)，可能导致模型偏向拒绝")
            print("  建议: 检查是否过于严格，适当放宽标注标准")
        elif merge_ratio > 70:
            print(f"\n  ⚠️  警告: 可合并样本过多 ({merge_ratio:.1f}%)，可能导致风险遗漏")
            print("  建议: 检查是否过于宽松，提高标注标准")
        else:
            print(f"\n  ✓ 样本比例合理 ({merge_ratio:.1f}%)，可以开始训练")
        
        # 训练建议
        if labeled_count < 150:
            print(f"\n  ⚠️  已标注数据较少 ({labeled_count} 条)，建议至少标注 150 条")
        elif labeled_count >= 180:
            print(f"\n  ✓ 数据量充足 ({labeled_count} 条)，可以训练高质量模型")
    
    if n_unlabeled > 0:
        print(f"\n下一步:")
        if labeled_count == 0:
            print(f"  1. 运行 python auto_label.py 自动标注约 85% 数据")
            print(f"  2. 运行 python 1_data_labeling.py 标注剩余数据")
        else:
            print(f"  - 还需标注 {n_unlabeled} 条数据")
            print(f"  - 运行 python 1_data_labeling.py 继续标注")
    else:
        print(f"\n✓ 所有数据已标注完成！可以开始训练模型")
        print(f"  - 运行 python 2_train_model.py 训练模型")
    
    # 显示未标注数据的索引（不超过20条）
    if 0 < n_unlabeled <= 20:
        unlabeled_idx = df[df[TARGET] == -1].index.tolist()
        print(f"\n未标注数据的行号（从 1 开始）:")
        print(f"  {[i+1 for i in unlabeled_idx]}")
    elif n_unlabeled > 20:
        unlabeled_idx = df[df[TARGET] == -1].index.tolist()[:20]
        print(f"\n前 20 条未标注数据的行号（从 1 开始）:")
        print(f"  {[i+1 for i in unlabeled_idx]} ...")


if __name__ == '__main__':
    check_labels()
