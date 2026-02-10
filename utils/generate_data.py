"""
统一的原始数据生成工具
可以控制生成数量
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import *

def generate_mr_data(n_samples=500, random_seed=None):
    """
    生成 MR 原始数据（未标注）
    
    参数:
        n_samples: 生成样本数量，默认500
        random_seed: 随机种子，None表示每次不同
    
    返回:
        DataFrame: 包含所有特征的数据，can_auto_merge=-1（未标注）
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed(None)
    
    print(f"生成 {n_samples} 条 MR 数据...")
    
    # 基本信息
    data = {
        'mr_id': [f'MR-{i+1:05d}' for i in range(n_samples)],
        'mr_title': [f'Feature/Fix #{i+1}' for i in range(n_samples)],
        'author': [f'author_{np.random.randint(1, 25)}' for _ in range(n_samples)],
        'repo': [f'repo_{np.random.randint(1, 12)}' for _ in range(n_samples)],
    }
    
    # 生成所有特征
    for feature in ALL_FEATURES:
        # === 代码变更规模 ===
        if feature == 'total_changed_lines':
            data[feature] = np.random.exponential(scale=40, size=n_samples).astype(int)
            data[feature] = np.clip(data[feature], 1, 1000)
        elif feature in ['added_lines', 'deleted_lines']:
            data[feature] = np.random.exponential(scale=25, size=n_samples).astype(int)
            data[feature] = np.clip(data[feature], 0, 600)
        elif feature == 'changed_files_count':
            data[feature] = np.random.poisson(lam=3.5, size=n_samples)
            data[feature] = np.clip(data[feature], 1, 30)
        elif feature == 'max_file_changed_lines':
            data[feature] = np.random.exponential(scale=35, size=n_samples).astype(int)
            data[feature] = np.clip(data[feature], 5, 500)
        
        # === 文件类型 ===
        elif feature in ['code_files_count', 'test_files_count']:
            if 'code' in feature:
                data[feature] = np.random.poisson(lam=2.5, size=n_samples)
            else:
                data[feature] = np.random.poisson(lam=1.2, size=n_samples)
            data[feature] = np.clip(data[feature], 0, 20)
        elif feature == 'code_lines_ratio':
            data[feature] = np.random.beta(6, 3, n_samples)
        elif feature in ['has_only_test_change', 'has_only_doc_change']:
            prob = 0.12 if 'test' in feature else 0.08
            data[feature] = np.random.choice([0, 1], size=n_samples, p=[1-prob, prob])
        
        # === 代码复杂度 ===
        elif feature == 'modified_methods_count':
            data[feature] = np.random.poisson(lam=4, size=n_samples)
            data[feature] = np.clip(data[feature], 0, 40)
        elif feature == 'max_method_complexity':
            data[feature] = np.random.triangular(left=1, mode=5, right=15, size=n_samples)
        elif feature == 'max_method_lines':
            data[feature] = np.random.exponential(scale=35, size=n_samples).astype(int)
            data[feature] = np.clip(data[feature], 10, 150)
        elif feature == 'max_nesting_level':
            data[feature] = np.random.poisson(lam=3, size=n_samples)
            data[feature] = np.clip(data[feature], 1, 8)
        
        # === 关键变更 ===
        elif feature in CRITICAL_CHANGES:
            if 'new_dependency' in feature:
                prob = 0.10
            elif 'database' in feature:
                prob = 0.05
            elif 'api' in feature:
                prob = 0.10
            elif 'breaking' in feature:
                prob = 0.06
            else:
                prob = 0.07
            data[feature] = np.random.choice([0, 1], size=n_samples, p=[1-prob, prob])
        
        # === 测试质量 ===
        elif feature == 'has_unit_test':
            data[feature] = np.random.choice([0, 1], size=n_samples, p=[0.35, 0.65])
        elif feature == 'unit_test_coverage':
            data[feature] = np.random.beta(5, 3, n_samples) * 100
            data[feature] = np.clip(data[feature], 0, 100)
        elif feature == 'test_coverage_change':
            data[feature] = np.random.normal(loc=0, scale=6, size=n_samples)
            data[feature] = np.clip(data[feature], -30, 30)
        elif feature == 'new_test_count':
            data[feature] = np.random.poisson(lam=2, size=n_samples)
            data[feature] = np.clip(data[feature], 0, 20)
        elif feature == 'test_to_code_ratio':
            data[feature] = np.random.beta(2, 4, n_samples)
        
        # === CI/CD 质量 ===
        elif feature in ['ci_build_passed', 'ci_unit_test_passed', 'ci_lint_passed']:
            if 'build' in feature or 'unit_test' in feature:
                prob = 0.88
            else:
                prob = 0.78
            data[feature] = np.random.choice([0, 1], size=n_samples, p=[1-prob, prob])
        elif feature in ['lint_issues_count', 'security_vulnerabilities', 'code_smells_count']:
            if 'lint' in feature:
                lam = 3.5
            elif 'security' in feature:
                lam = 0.3
            else:
                lam = 2.5
            data[feature] = np.random.poisson(lam=lam, size=n_samples)
            data[feature] = np.clip(data[feature], 0, 50)
        
        # === 作者质量 ===
        elif feature == 'author_total_mrs':
            data[feature] = np.random.lognormal(mean=3.2, sigma=1.3, size=n_samples).astype(int)
            data[feature] = np.clip(data[feature], 1, 500)
        elif feature == 'author_mrs_last_30d':
            data[feature] = np.random.poisson(lam=4, size=n_samples)
            data[feature] = np.clip(data[feature], 0, 40)
        elif feature == 'author_merge_rate_30d':
            data[feature] = np.random.beta(7, 2, n_samples)
        elif feature == 'author_revert_rate_30d':
            data[feature] = np.random.beta(1, 15, n_samples)
        elif feature == 'author_days_in_repo':
            data[feature] = np.random.lognormal(mean=4.5, sigma=1.5, size=n_samples).astype(int)
            data[feature] = np.clip(data[feature], 1, 2000)
        elif feature in ['is_first_mr', 'is_core_contributor']:
            prob = 0.10 if 'first' in feature else 0.22
            data[feature] = np.random.choice([0, 1], size=n_samples, p=[1-prob, prob])
        elif feature == 'author_avg_mr_size':
            data[feature] = np.random.lognormal(mean=4, sigma=1.2, size=n_samples).astype(int)
            data[feature] = np.clip(data[feature], 10, 800)
        elif feature == 'author_bug_fix_count_30d':
            data[feature] = np.random.poisson(lam=2, size=n_samples)
            data[feature] = np.clip(data[feature], 0, 25)
        
        # === 提交质量 ===
        elif feature == 'commit_count':
            data[feature] = np.random.poisson(lam=3, size=n_samples)
            data[feature] = np.clip(data[feature], 1, 20)
        elif feature == 'avg_commit_size':
            data[feature] = np.random.exponential(scale=45, size=n_samples).astype(int)
            data[feature] = np.clip(data[feature], 10, 300)
        elif feature == 'commit_message_quality':
            data[feature] = np.random.triangular(left=4, mode=7, right=10, size=n_samples)
        
        # === 仓库权限 ===
        elif feature == 'repo_importance_score':
            data[feature] = np.random.triangular(left=3, mode=6, right=10, size=n_samples)
        elif feature in ['is_repo_owner', 'is_maintainer']:
            prob = 0.08 if 'owner' in feature else 0.18
            data[feature] = np.random.choice([0, 1], size=n_samples, p=[1-prob, prob])
        
        # === 风险信号 ===
        elif feature in RISK_SIGNALS:
            if feature == 'involves_payment':
                prob = 0.02
            elif feature == 'involves_pii':
                prob = 0.04
            elif feature == 'involves_auth':
                prob = 0.06
            elif feature == 'involves_security':
                prob = 0.05
            elif feature == 'modifies_critical_file':
                prob = 0.08
            elif feature == 'has_todo_or_fixme':
                prob = 0.15
            else:
                prob = 0.05
            data[feature] = np.random.choice([0, 1], size=n_samples, p=[1-prob, prob])
        
        else:
            data[feature] = 0
    
    # 初始化目标变量为未标注
    data[TARGET] = [-1] * n_samples
    
    df = pd.DataFrame(data)
    
    print(f"[OK] 生成完成: {len(df)} 条数据，{len(ALL_FEATURES)} 个特征")
    
    return df


def save_data(df, output_path):
    """保存数据到CSV"""
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 已保存到: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("MR 数据生成工具")
    print("=" * 80)
    
    # 询问生成数量
    try:
        n = input("\n生成多少条数据？(默认 500): ").strip()
        n_samples = int(n) if n else 500
    except:
        n_samples = 500
    
    # 生成数据
    df = generate_mr_data(n_samples=n_samples, random_seed=42)
    
    # 保存到 data 目录
    output_path = RAW_DATA_PATH
    save_data(df, output_path)
    
    print("\n" + "=" * 80)
    print("数据生成完成！")
    print("=" * 80)
    print(f"\n输出文件: {output_path}")
    print(f"\n下一步:")
    print(f"  1. 运行: python utils/auto_label.py 自动标注")
    print(f"  2. 运行: python 1_data_labeling.py 人工标注剩余数据")


if __name__ == '__main__':
    main()
