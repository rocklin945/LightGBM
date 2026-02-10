"""
测试集生成工具
组合数据生成和自动标注功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from utils.generate_data import generate_mr_data
from utils.auto_label import auto_label_dataframe, print_statistics

def create_test_set(n_samples=100, random_seed=None):
    """
    创建测试集
    
    参数:
        n_samples: 测试集样本数量
        random_seed: 随机种子
    
    返回:
        df: 标注后的测试集DataFrame
        stats: 标注统计信息
    """
    print(f"生成 {n_samples} 条测试数据...")
    
    # 1. 生成数据
    df = generate_mr_data(n_samples=n_samples, random_seed=random_seed)
    
    # 2. 自动标注
    print("\n自动标注测试数据...")
    df, stats = auto_label_dataframe(df, skip_labeled=False)
    
    return df, stats


def main():
    """主函数"""
    print("=" * 80)
    print("测试集生成工具")
    print("=" * 80)
    
    # 询问生成数量
    try:
        n = input("\n生成多少条测试数据？(默认 100): ").strip()
        n_samples = int(n) if n else 100
    except:
        n_samples = 100
    
    # 生成测试集
    df, stats = create_test_set(n_samples=n_samples, random_seed=None)
    
    # 打印统计
    print_statistics(stats, len(df))
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/test_sets/test_{timestamp}.csv"
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 已保存到: {output_path}")
    
    print("\n" + "=" * 80)
    print("测试集生成完成！")
    print("=" * 80)
    print(f"\n输出文件: {output_path}")
    print(f"样本数量: {len(df)}")
    print(f"标注完成: {stats['can_merge'] + stats['cannot_merge']} 条")
    print(f"未标注:   {stats['unlabeled']} 条")
    
    if stats['unlabeled'] > 0:
        print(f"\n注意: 还有 {stats['unlabeled']} 条数据未能自动标注")
        print(f"如需完整测试集，请手动标注")
    
    print(f"\n下一步:")
    print(f"  运行: python models/comparison/test_all_models.py")


if __name__ == '__main__':
    main()
