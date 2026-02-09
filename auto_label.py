"""
自动标注脚本（改进版）
基于规则自动标注，确保正负样本平衡
"""

import pandas as pd
import numpy as np
from config import *

class AutoLabeler:
    def __init__(self):
        """初始化自动标注器"""
        self.rules = []
        self.setup_rules()
    
    def setup_rules(self):
        """设置标注规则（更平衡的规则）"""
        
        # === 规则 1: 明确可以自动合并的情况 ===
        def rule_safe_merge(row):
            """安全可合并"""
            # 1.1 纯文档改动
            if row['has_only_doc_change'] == 1:
                return 1, "纯文档改动"
            
            # 1.2 纯测试改动 + 测试通过
            if row['has_only_test_change'] == 1 and row['ci_unit_test_passed'] == 1:
                return 1, "纯测试改动且通过"
            
            # 1.3 Owner/Maintainer + 小改动 + CI全通过
            if (row['is_repo_owner'] == 1 or row['is_maintainer'] == 1):
                if (row['total_changed_lines'] <= 30 and 
                    row['changed_files_count'] <= 2 and
                    row['ci_build_passed'] == 1 and
                    row['ci_unit_test_passed'] == 1):
                    return 1, "Owner/Maintainer小改动"
            
            # 1.4 核心贡献者 + 微小改动 + 全通过
            if row['is_core_contributor'] == 1:
                if (row['total_changed_lines'] <= 50 and
                    row['changed_files_count'] <= 3 and
                    row['ci_build_passed'] == 1 and
                    row['ci_unit_test_passed'] == 1 and
                    row['ci_lint_passed'] == 1):
                    return 1, "核心贡献者微小改动"
            
            # 1.5 资深作者 + 小改动 + 高测试覆盖
            if (row['author_total_mrs'] >= 50 and
                row['author_merge_rate_30d'] >= 0.85 and
                row['author_revert_rate_30d'] <= 0.03):
                if (row['total_changed_lines'] <= 100 and
                    row['changed_files_count'] <= 5 and
                    row['ci_build_passed'] == 1 and
                    row['has_unit_test'] == 1 and
                    row['unit_test_coverage'] >= 70):
                    return 1, "资深作者+小改动+高质量"
            
            return None, None
        
        # === 规则 2: 明确不可以自动合并的情况 ===
        def rule_must_reject(row):
            """必须拒绝"""
            # 2.1 强制拒绝条件
            if row['involves_payment'] == 1:
                return 0, "涉及支付"
            if row['ci_build_passed'] == 0:
                return 0, "CI构建失败"
            
            # 2.2 高风险变更
            if row['involves_pii'] == 1:
                return 0, "涉及个人信息"
            if row['has_breaking_change'] == 1:
                return 0, "破坏性变更"
            if row['has_database_migration'] == 1:
                return 0, "数据库迁移"
            if row['security_vulnerabilities'] > 0:
                return 0, "存在安全漏洞"
            
            # 2.3 大改动
            if row['total_changed_lines'] > 500:
                return 0, "改动过大(>500行)"
            if row['changed_files_count'] > 15:
                return 0, "改动文件过多(>15)"
            
            # 2.4 新手 + 非小改动
            if row['is_first_mr'] == 1 and row['total_changed_lines'] > 50:
                return 0, "首次MR且改动较大"
            if row['author_total_mrs'] < 3:
                return 0, "作者经验不足(<3个MR)"
            
            # 2.5 测试问题
            if row['ci_unit_test_passed'] == 0:
                return 0, "单元测试失败"
            if row['code_lines_ratio'] > 0.7 and row['has_unit_test'] == 0:
                return 0, "代码改动无单元测试"
            if row['test_coverage_change'] < -15:
                return 0, "测试覆盖率大幅下降"
            
            # 2.6 代码质量问题
            if row['has_todo_or_fixme'] == 1:
                return 0, "代码有TODO/FIXME"
            if row['lint_issues_count'] > 15:
                return 0, "代码规范问题过多"
            if row['code_smells_count'] > 15:
                return 0, "代码坏味道过多"
            
            return None, None
        
        # === 规则 3: 中等风险（倾向拒绝）===
        def rule_medium_risk_reject(row):
            """中等风险倾向拒绝"""
            # 3.1 有风险信号
            if row['involves_auth'] == 1 or row['involves_security'] == 1:
                return 0, "涉及权限或安全"
            if row['modifies_critical_file'] == 1:
                return 0, "修改关键文件"
            
            # 3.2 有API变更且非Owner/Maintainer
            if row['has_api_change'] == 1:
                if row['is_maintainer'] == 0:
                    return 0, "非Maintainer的API变更"
            
            # 3.3 较大改动 + 测试不足
            if row['total_changed_lines'] > 200:
                if row['unit_test_coverage'] < 60:
                    return 0, "大改动+测试覆盖不足"
            
            # 3.4 新依赖 + 非核心贡献者
            if row['has_new_dependency'] == 1:
                if row['is_core_contributor'] == 0:
                    return 0, "非核心贡献者添加依赖"
            
            return None, None
        
        # === 规则 4: 中等风险（倾向通过）===
        def rule_medium_risk_pass(row):
            """中等风险倾向通过"""
            # 4.1 有经验的作者 + 中等改动 + CI通过
            if (row['author_total_mrs'] >= 10 and
                row['author_merge_rate_30d'] >= 0.75):
                if (row['total_changed_lines'] <= 300 and
                    row['changed_files_count'] <= 8 and
                    row['ci_build_passed'] == 1 and
                    row['ci_unit_test_passed'] == 1 and
                    row['ci_lint_passed'] == 1):
                    return 1, "有经验作者+中等改动+CI全过"
            
            return None, None
        
        # 按优先级排序
        self.rules = [
            rule_safe_merge,           # 1. 明确安全（优先标注为1）
            rule_must_reject,          # 2. 明确拒绝（优先标注为0）
            rule_medium_risk_reject,   # 3. 中等风险倾向拒绝
            rule_medium_risk_pass,     # 4. 中等风险倾向通过
        ]
    
    def label_row(self, row):
        """对单行数据应用规则"""
        for rule in self.rules:
            label, reason = rule(row)
            if label is not None:
                return label, reason
        
        # 如果所有规则都不匹配，返回不确定
        return -1, "未匹配任何规则"
    
    def auto_label_data(self, input_file, output_file):
        """自动标注数据"""
        print("=" * 80)
        print("自动标注工具（改进版）")
        print("=" * 80)
        
        # 读取数据
        print(f"\n读取数据: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        print(f"✓ 已加载 {len(df)} 条记录")
        
        # 统计现有标注
        labeled_count = len(df[df[TARGET] != -1])
        print(f"  - 已标注: {labeled_count}")
        print(f"  - 未标注: {len(df) - labeled_count}")
        
        # 自动标注
        print("\n开始自动标注...")
        auto_labeled = 0
        uncertain = 0
        reasons = []
        
        for idx, row in df.iterrows():
            # 跳过已标注的
            if row[TARGET] != -1:
                continue
            
            label, reason = self.label_row(row)
            
            if label != -1:
                df.at[idx, TARGET] = label
                auto_labeled += 1
                reasons.append((label, reason))
            else:
                uncertain += 1
        
        print(f"\n✓ 自动标注完成")
        print(f"  - 自动标注: {auto_labeled}")
        print(f"  - 仍需人工: {uncertain}")
        
        # 统计标注原因
        if reasons:
            print("\n标注原因统计:")
            from collections import Counter
            
            can_merge_reasons = [r for l, r in reasons if l == 1]
            cannot_merge_reasons = [r for l, r in reasons if l == 0]
            
            if can_merge_reasons:
                print("\n  【可以自动合并的原因】")
                for reason, count in Counter(can_merge_reasons).most_common(5):
                    print(f"    {reason:45s}: {count:3d}")
            
            if cannot_merge_reasons:
                print("\n  【不可以自动合并的原因】")
                for reason, count in Counter(cannot_merge_reasons).most_common(5):
                    print(f"    {reason:45s}: {count:3d}")
        
        # 统计结果
        n_can_merge = len(df[df[TARGET] == 1])
        n_cannot_merge = len(df[df[TARGET] == 0])
        n_unlabeled = len(df[df[TARGET] == -1])
        
        print("\n" + "=" * 80)
        print("最终统计")
        print("=" * 80)
        print(f"  可以自动合并:   {n_can_merge:4d} ({n_can_merge/len(df)*100:5.1f}%)")
        print(f"  不可以自动合并: {n_cannot_merge:4d} ({n_cannot_merge/len(df)*100:5.1f}%)")
        print(f"  仍需人工标注:   {n_unlabeled:4d} ({n_unlabeled/len(df)*100:5.1f}%)")
        
        if n_can_merge + n_cannot_merge > 0:
            merge_ratio = n_can_merge / (n_can_merge + n_cannot_merge) * 100
            print(f"\n  已标注数据中自动合并比例: {merge_ratio:.1f}%")
            
            # 检查样本平衡性
            if merge_ratio < 30:
                print(f"\n  ⚠️  警告: 可合并样本过少 ({merge_ratio:.1f}%)，建议调整规则")
            elif merge_ratio > 70:
                print(f"\n  ⚠️  警告: 可合并样本过多 ({merge_ratio:.1f}%)，规则可能太宽松")
            else:
                print(f"\n  ✓ 样本比例合理 ({merge_ratio:.1f}%)")
        
        # 保存
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ 已保存到: {output_file}")
        
        if n_unlabeled > 0:
            print(f"\n提示: 还有 {n_unlabeled} 条数据需要人工标注")
            print(f"运行: python 1_data_labeling.py")
        else:
            print(f"\n✓ 所有数据已标注完成，可以开始训练")
            print(f"运行: python 2_train_model.py")


def main():
    """主函数"""
    labeler = AutoLabeler()
    labeler.auto_label_data(RAW_DATA_PATH, RAW_DATA_PATH)
    
    # 保存到 labeled_data.csv
    df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')
    labeled_df = df[df[TARGET] != -1]
    if len(labeled_df) > 0:
        labeled_df.to_csv(LABELED_DATA_PATH, index=False, encoding='utf-8-sig')
        print(f"✓ 已保存 {len(labeled_df)} 条已标注数据到: {LABELED_DATA_PATH}")


if __name__ == '__main__':
    main()
