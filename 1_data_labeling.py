"""
MR 数据标注工具（精简版）
50个核心特征，200条样本
"""

import pandas as pd
import numpy as np
import os
from config import *

class MRDataLabeler:
    def __init__(self):
        """初始化数据标注工具"""
        self.ensure_data_dir()
        self.df = None
        self.current_index = 0
        
    def ensure_data_dir(self):
        """确保数据目录存在"""
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def create_sample_data(self, n_samples=200):
        """创建示例数据（200条，50个特征）"""
        print(f"创建 {n_samples} 条示例数据（{len(ALL_FEATURES)} 个特征）...")
        
        np.random.seed(42)
        
        data = {
            'mr_id': [f'MR-{i+1:05d}' for i in range(n_samples)],
            'mr_title': [f'Feature/Fix #{i+1}' for i in range(n_samples)],
            'author': [f'author_{np.random.randint(1, 25)}' for _ in range(n_samples)],
            'repo': [f'repo_{np.random.randint(1, 12)}' for _ in range(n_samples)],
        }
        
        # 生成特征值（按类别）
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
                # 默认（不应该到这里）
                print(f"警告: 未处理特征 {feature}")
                data[feature] = 0
        
        # 初始化目标变量
        data[TARGET] = [-1] * n_samples
        
        df = pd.DataFrame(data)
        df.to_csv(RAW_DATA_PATH, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存: {RAW_DATA_PATH}")
        print(f"  - 样本数: {len(df)}")
        print(f"  - 特征数: {len(ALL_FEATURES)}")
        
        # 显示统计
        self.show_data_statistics(df)
        
        return df
    
    def show_data_statistics(self, df):
        """显示数据统计"""
        print("\n" + "=" * 70)
        print("数据统计")
        print("=" * 70)
        
        print("\n【风险信号分布】")
        for feature in RISK_SIGNALS:
            count = df[feature].sum()
            pct = count / len(df) * 100
            print(f"  {FEATURE_DESCRIPTIONS[feature]:30s}: {count:3d} ({pct:5.1f}%)")
        
        print("\n【CI质量分布】")
        ci_features = ['ci_build_passed', 'ci_unit_test_passed', 'ci_lint_passed']
        for feature in ci_features:
            count = df[feature].sum()
            pct = count / len(df) * 100
            print(f"  {FEATURE_DESCRIPTIONS[feature]:30s}: {count:3d} ({pct:5.1f}%)")
        
        print("\n【关键数值统计】")
        key_features = ['total_changed_lines', 'changed_files_count', 
                       'author_total_mrs', 'unit_test_coverage']
        for feature in key_features:
            mean_val = df[feature].mean()
            median_val = df[feature].median()
            print(f"  {FEATURE_DESCRIPTIONS[feature]:30s}: 均值={mean_val:6.1f}, 中位数={median_val:6.1f}")
    
    def load_data(self):
        """加载数据"""
        if not os.path.exists(RAW_DATA_PATH):
            print(f"数据文件不存在，创建示例数据...")
            self.df = self.create_sample_data()
        else:
            self.df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')
            print(f"已加载数据: {len(self.df)} 条记录")
            
            # 检查特征匹配
            existing_features = set(self.df.columns) - {'mr_id', 'mr_title', 'author', 'repo', TARGET}
            required_features = set(ALL_FEATURES)
            
            if existing_features != required_features:
                print("\n⚠️  特征不匹配！")
                print(f"  数据文件: {len(existing_features)} 个特征")
                print(f"  当前配置: {len(required_features)} 个特征")
                
                choice = input("\n删除旧数据重新生成？(y/n): ").strip().lower()
                if choice == 'y':
                    if os.path.exists(RAW_DATA_PATH):
                        os.remove(RAW_DATA_PATH)
                    if os.path.exists(LABELED_DATA_PATH):
                        os.remove(LABELED_DATA_PATH)
                    self.df = self.create_sample_data()
                else:
                    exit(0)
        
        # 找第一条未标注的
        unlabeled = self.df[self.df[TARGET] == -1]
        if len(unlabeled) > 0:
            self.current_index = unlabeled.index[0]
        else:
            self.current_index = 0
    
    def show_current_mr(self):
        """显示当前MR"""
        if self.current_index >= len(self.df):
            return False
        
        row = self.df.iloc[self.current_index]
        
        print("\n" + "=" * 80)
        print(f"进度: {self.current_index + 1}/{len(self.df)}")
        print("=" * 80)
        print(f"MR ID: {row['mr_id']} | 作者: {row['author']} | 仓库: {row['repo']}")
        print("-" * 80)
        
        # 关键特征快速预览
        print("\n【关键指标】")
        print(f"  改动: {row['total_changed_lines']:.0f} 行, {row['changed_files_count']:.0f} 文件")
        print(f"  CI: 构建{'✓' if row['ci_build_passed']==1 else '✗'} | 测试{'✓' if row['ci_unit_test_passed']==1 else '✗'} | 规范{'✓' if row['ci_lint_passed']==1 else '✗'}")
        print(f"  测试: {'有' if row['has_unit_test']==1 else '无'}单测, 覆盖率 {row['unit_test_coverage']:.0f}%")
        print(f"  作者: {row['author_total_mrs']:.0f} 个MR, 合并率 {row['author_merge_rate_30d']:.1%}")
        
        # 风险信号
        risks = []
        for feature in RISK_SIGNALS:
            if row[feature] == 1:
                risks.append(FEATURE_DESCRIPTIONS[feature])
        
        if risks:
            print(f"\n  ⚠️  风险信号: {', '.join(risks)}")
        
        # 判断建议
        print("\n【AI建议】", end=" ")
        
        # 简单规则判断
        if row['involves_payment'] == 1 or row['ci_build_passed'] == 0:
            print("❌ 不可自动合并（强制拒绝）")
        elif row['has_only_doc_change'] == 1 or row['has_only_test_change'] == 1:
            print("✅ 可以自动合并（安全改动）")
        elif (row['is_repo_owner'] == 1 or row['is_maintainer'] == 1) and row['total_changed_lines'] <= 50:
            print("✅ 可以自动合并（Owner/Maintainer小改动）")
        elif row['total_changed_lines'] > 500 or row['is_first_mr'] == 1:
            print("❌ 不可自动合并（大改动或新手）")
        else:
            print("❓ 需要判断")
        
        print("\n按 'd' 显示所有特征")
        
        return True
    
    def label_current_mr(self):
        """标注当前MR"""
        while True:
            choice = input("\n标注 (1=可合并, 0=不可合并, s=跳过, q=退出, d=详情): ").strip().lower()
            
            if choice == 'q':
                return 'quit'
            elif choice == 's':
                self.current_index += 1
                return 'skip'
            elif choice == 'd':
                self.show_all_features()
                continue
            elif choice in ['0', '1']:
                label = int(choice)
                self.df.at[self.current_index, TARGET] = label
                self.current_index += 1
                print(f"✓ 已标注为: {'可以自动合并' if label == 1 else '不可以自动合并'}")
                return 'labeled'
            else:
                print("无效输入")
    
    def show_all_features(self):
        """显示所有特征"""
        row = self.df.iloc[self.current_index]
        print("\n" + "=" * 80)
        print(f"完整特征列表 - MR ID: {row['mr_id']}")
        print("=" * 80)
        
        categories = [
            ('代码变更规模', CODE_CHANGE_SIZE),
            ('文件类型', FILE_TYPE_FEATURES),
            ('代码复杂度', CODE_COMPLEXITY),
            ('关键变更', CRITICAL_CHANGES),
            ('测试质量', TEST_QUALITY),
            ('CI/CD质量', CI_QUALITY),
            ('作者质量', AUTHOR_QUALITY),
            ('提交质量', COMMIT_QUALITY),
            ('仓库权限', REPO_PERMISSION),
            ('风险信号', RISK_SIGNALS),
        ]
        
        for cat_name, features in categories:
            print(f"\n【{cat_name}】")
            for feature in features:
                value = row[feature]
                desc = FEATURE_DESCRIPTIONS[feature]
                if isinstance(value, float) and value < 1:
                    print(f"  {desc:25s}: {value:.3f}")
                elif isinstance(value, float):
                    print(f"  {desc:25s}: {value:.1f}")
                else:
                    print(f"  {desc:25s}: {value}")
        
        input("\n按 Enter 继续...")
    
    def save_data(self):
        """保存数据"""
        self.df.to_csv(RAW_DATA_PATH, index=False, encoding='utf-8-sig')
        
        labeled_df = self.df[self.df[TARGET] != -1]
        if len(labeled_df) > 0:
            labeled_df.to_csv(LABELED_DATA_PATH, index=False, encoding='utf-8-sig')
        
        n_can = len(self.df[self.df[TARGET] == 1])
        n_cannot = len(self.df[self.df[TARGET] == 0])
        n_unlabeled = len(self.df[self.df[TARGET] == -1])
        
        print("\n" + "=" * 70)
        print("标注统计")
        print("=" * 70)
        print(f"  可以自动合并:   {n_can:4d}")
        print(f"  不可以自动合并: {n_cannot:4d}")
        print(f"  未标注:         {n_unlabeled:4d}")
        
        if n_can + n_cannot > 0:
            print(f"  自动合并比例:   {n_can/(n_can + n_cannot)*100:.1f}%")
    
    def run(self):
        """运行标注"""
        print("=" * 80)
        print("MR 数据标注工具")
        print("=" * 80)
        
        self.load_data()
        
        n_can = len(self.df[self.df[TARGET] == 1])
        n_cannot = len(self.df[self.df[TARGET] == 0])
        n_unlabeled = len(self.df[self.df[TARGET] == -1])
        
        print(f"\n当前进度: 已标注 {n_can + n_cannot}/{len(self.df)}, 剩余 {n_unlabeled}")
        
        if n_unlabeled == 0:
            print("\n✓ 所有数据已标注完成！")
            print("\n1 - 退出 | 2 - 查看统计 | 3 - 重新标注")
            choice = input("选择: ").strip()
            if choice == '3':
                self.df[TARGET] = -1
                self.current_index = 0
            else:
                return
        else:
            input(f"\n从第 {self.current_index + 1} 条开始，按 Enter 继续...")
        
        while self.current_index < len(self.df):
            # 跳过已标注
            while self.current_index < len(self.df) and self.df.iloc[self.current_index][TARGET] != -1:
                self.current_index += 1
            
            if self.current_index >= len(self.df):
                break
            
            if not self.show_current_mr():
                break
            
            if self.label_current_mr() == 'quit':
                break
        
        self.save_data()
        print("\n✓ 标注完成！")


def main():
    labeler = MRDataLabeler()
    
    if not os.path.exists(RAW_DATA_PATH):
        print("首次运行，创建200条示例数据...")
        labeler.create_sample_data(n_samples=200)
        print("\n✓ 数据创建完成！")
        print("\n建议: 先运行 python auto_label.py 自动标注大部分数据")
        print("然后再运行 python 1_data_labeling.py 标注剩余数据\n")
        return
    
    labeler.run()


if __name__ == '__main__':
    main()
