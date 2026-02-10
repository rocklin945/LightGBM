"""
全自动标注工具
使用规则和评分系统自动标注所有数据，无需人工介入
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import *
from collections import Counter

def auto_label_rules(row):
    """
    自动标注规则
    返回: (label, reason) 或 (None, None)
    """
    # === 规则1: 明确可以自动合并 ===
    if row['has_only_doc_change'] == 1:
        return 1, "纯文档改动"
    
    if row['has_only_test_change'] == 1 and row['ci_unit_test_passed'] == 1:
        return 1, "纯测试改动且通过"
    
    if (row['is_repo_owner'] == 1 or row['is_maintainer'] == 1):
        if (row['total_changed_lines'] <= 30 and 
            row['changed_files_count'] <= 2 and
            row['ci_build_passed'] == 1 and
            row['ci_unit_test_passed'] == 1):
            return 1, "Owner/Maintainer小改动"
    
    if row['is_core_contributor'] == 1:
        if (row['total_changed_lines'] <= 50 and
            row['changed_files_count'] <= 3 and
            row['ci_build_passed'] == 1 and
            row['ci_unit_test_passed'] == 1 and
            row['ci_lint_passed'] == 1):
            return 1, "核心贡献者微小改动"
    
    if (row['author_total_mrs'] >= 50 and
        row['author_merge_rate_30d'] >= 0.85 and
        row['author_revert_rate_30d'] <= 0.03):
        if (row['total_changed_lines'] <= 100 and
            row['changed_files_count'] <= 5 and
            row['ci_build_passed'] == 1 and
            row['has_unit_test'] == 1 and
            row['unit_test_coverage'] >= 70):
            return 1, "资深作者+小改动+高质量"
    
    # === 规则2: 必须拒绝 ===
    if row['involves_payment'] == 1:
        return 0, "涉及支付"
    if row['ci_build_passed'] == 0:
        return 0, "CI构建失败"
    if row['involves_pii'] == 1:
        return 0, "涉及个人信息"
    if row['has_breaking_change'] == 1:
        return 0, "破坏性变更"
    if row['has_database_migration'] == 1:
        return 0, "数据库迁移"
    if row['security_vulnerabilities'] > 0:
        return 0, "存在安全漏洞"
    if row['total_changed_lines'] > 500:
        return 0, "改动过大(>500行)"
    if row['changed_files_count'] > 15:
        return 0, "改动文件过多(>15)"
    if row['is_first_mr'] == 1 and row['total_changed_lines'] > 50:
        return 0, "首次MR且改动较大"
    if row['author_total_mrs'] < 3:
        return 0, "作者经验不足(<3个MR)"
    if row['ci_unit_test_passed'] == 0:
        return 0, "单元测试失败"
    if row['code_lines_ratio'] > 0.7 and row['has_unit_test'] == 0:
        return 0, "代码改动无单元测试"
    if row['test_coverage_change'] < -15:
        return 0, "测试覆盖率大幅下降"
    if row['has_todo_or_fixme'] == 1:
        return 0, "代码有TODO/FIXME"
    if row['lint_issues_count'] > 15:
        return 0, "代码规范问题过多"
    if row['code_smells_count'] > 15:
        return 0, "代码坏味道过多"
    
    # === 规则3: 中等风险倾向拒绝 ===
    if row['involves_auth'] == 1 or row['involves_security'] == 1:
        return 0, "涉及权限或安全"
    if row['modifies_critical_file'] == 1:
        return 0, "修改关键文件"
    if row['has_api_change'] == 1 and row['is_maintainer'] == 0:
        return 0, "非Maintainer的API变更"
    if row['total_changed_lines'] > 200 and row['unit_test_coverage'] < 60:
        return 0, "大改动+测试覆盖不足"
    if row['has_new_dependency'] == 1 and row['is_core_contributor'] == 0:
        return 0, "非核心贡献者添加依赖"
    
    # === 规则4: 中等风险倾向通过 ===
    if (row['author_total_mrs'] >= 10 and
        row['author_merge_rate_30d'] >= 0.75):
        if (row['total_changed_lines'] <= 300 and
            row['changed_files_count'] <= 8 and
            row['ci_build_passed'] == 1 and
            row['ci_unit_test_passed'] == 1 and
            row['ci_lint_passed'] == 1):
            return 1, "有经验作者+中等改动+CI全过"
    
    # === 规则5: 兜底策略 - 基于评分自动判断 ===
    # 如果以上规则都无法判断，使用评分系统
    score = calculate_merge_score(row)
    
    # 评分 >= 60 认为可以合并，< 60 认为不可以合并
    if score >= 60:
        return 1, f"评分通过(分数:{score:.0f})"
    else:
        return 0, f"评分不足(分数:{score:.0f})"


def calculate_merge_score(row):
    """
    计算合并评分（0-100分）
    用于无法通过明确规则判断的情况
    """
    score = 50  # 基础分
    
    # 1. 作者质量 (最多 +20分)
    if row['author_total_mrs'] >= 20:
        score += 10
    elif row['author_total_mrs'] >= 10:
        score += 5
    
    if row['author_merge_rate_30d'] >= 0.7:
        score += 5
    
    if row['author_revert_rate_30d'] <= 0.05:
        score += 5
    
    # 2. 代码规模 (改动越小越好，最多 +15分)
    if row['total_changed_lines'] <= 50:
        score += 15
    elif row['total_changed_lines'] <= 100:
        score += 10
    elif row['total_changed_lines'] <= 200:
        score += 5
    else:
        score -= 10  # 改动太大扣分
    
    if row['changed_files_count'] <= 3:
        score += 5
    elif row['changed_files_count'] > 10:
        score -= 5
    
    # 3. CI质量 (最多 +15分)
    if row['ci_build_passed'] == 1:
        score += 5
    else:
        score -= 15  # 构建失败严重扣分
    
    if row['ci_unit_test_passed'] == 1:
        score += 5
    else:
        score -= 10
    
    if row['ci_lint_passed'] == 1:
        score += 5
    
    # 4. 测试质量 (最多 +10分)
    if row['has_unit_test'] == 1:
        score += 5
    
    if row['unit_test_coverage'] >= 80:
        score += 5
    elif row['unit_test_coverage'] >= 60:
        score += 3
    
    # 5. 代码质量 (最多 +10分)
    if row['lint_issues_count'] == 0:
        score += 5
    elif row['lint_issues_count'] > 10:
        score -= 5
    
    if row['code_smells_count'] <= 5:
        score += 5
    elif row['code_smells_count'] > 10:
        score -= 5
    
    # 6. 风险因素 (扣分项)
    if row['security_vulnerabilities'] > 0:
        score -= 20
    
    # 限制分数范围
    score = max(0, min(100, score))
    
    return score


def auto_label_dataframe(df, skip_labeled=True):
    """
    对DataFrame进行全自动标注（所有数据都会被标注）
    
    参数:
        df: 输入DataFrame
        skip_labeled: 是否跳过已标注的数据
    
    返回:
        df: 标注后的DataFrame
        stats: 标注统计信息
    """
    rule_labeled = 0
    score_labeled = 0
    reasons = []
    
    for idx, row in df.iterrows():
        # 跳过已标注的
        if skip_labeled and row[TARGET] != -1:
            continue
        
        label, reason = auto_label_rules(row)
        
        # 所有数据都会被标注（通过规则或评分）
        df.at[idx, TARGET] = label
        reasons.append((label, reason))
        
        if '评分' in reason:
            score_labeled += 1
        else:
            rule_labeled += 1
    
    # 统计
    n_can = len(df[df[TARGET] == 1])
    n_cannot = len(df[df[TARGET] == 0])
    n_unlabeled = len(df[df[TARGET] == -1])
    
    stats = {
        'rule_labeled': rule_labeled,
        'score_labeled': score_labeled,
        'total_labeled': rule_labeled + score_labeled,
        'can_merge': n_can,
        'cannot_merge': n_cannot,
        'unlabeled': n_unlabeled,
        'reasons': reasons
    }
    
    return df, stats


def print_statistics(stats, total):
    """打印标注统计"""
    print("\n" + "=" * 80)
    print("全自动标注统计")
    print("=" * 80)
    
    print(f"\n本次标注方式:")
    print(f"  规则标注: {stats['rule_labeled']:4d} 条")
    print(f"  评分标注: {stats['score_labeled']:4d} 条")
    print(f"  总计:     {stats['total_labeled']:4d} 条")
    
    print(f"\n标注结果:")
    print(f"  可以自动合并:   {stats['can_merge']:4d} ({stats['can_merge']/total*100:5.1f}%)")
    print(f"  不可以自动合并: {stats['cannot_merge']:4d} ({stats['cannot_merge']/total*100:5.1f}%)")
    print(f"  未标注:         {stats['unlabeled']:4d} ({stats['unlabeled']/total*100:5.1f}%)")
    
    if stats['can_merge'] + stats['cannot_merge'] > 0:
        merge_ratio = stats['can_merge'] / (stats['can_merge'] + stats['cannot_merge']) * 100
        print(f"\n  自动合并比例: {merge_ratio:.1f}%")
        
        if merge_ratio < 30:
            print(f"  ⚠️  可合并样本较少，模型会更谨慎")
        elif merge_ratio > 70:
            print(f"  ⚠️  可合并样本较多，注意模型泛化")
        else:
            print(f"  ✓ 样本比例合理")
    
    # 标注原因统计
    if stats['reasons']:
        print("\n标注原因统计 (Top 5):")
        
        can_merge_reasons = [r for l, r in stats['reasons'] if l == 1]
        cannot_merge_reasons = [r for l, r in stats['reasons'] if l == 0]
        
        if can_merge_reasons:
            print("\n  【可以自动合并】")
            for reason, count in Counter(can_merge_reasons).most_common(5):
                print(f"    {reason:45s}: {count:3d}")
        
        if cannot_merge_reasons:
            print("\n  【不可以自动合并】")
            for reason, count in Counter(cannot_merge_reasons).most_common(5):
                print(f"    {reason:45s}: {count:3d}")


def main():
    """主函数"""
    print("=" * 80)
    print("全自动标注工具")
    print("=" * 80)
    
    # 检查数据文件
    if not os.path.exists(RAW_DATA_PATH):
        print(f"\n错误: 数据文件不存在 {RAW_DATA_PATH}")
        print(f"请先运行: python utils/generate_data.py")
        return
    
    # 加载数据
    print(f"\n读取数据: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')
    print(f"[OK] 已加载 {len(df)} 条记录")
    
    # 统计现有标注
    labeled_count = len(df[df[TARGET] != -1])
    unlabeled_count = len(df) - labeled_count
    print(f"  - 已标注: {labeled_count}")
    print(f"  - 待标注: {unlabeled_count}")
    
    if unlabeled_count == 0:
        print(f"\n所有数据已标注完成！")
        return
    
    # 全自动标注
    print(f"\n开始全自动标注 {unlabeled_count} 条数据...")
    df, stats = auto_label_dataframe(df, skip_labeled=True)
    
    # 打印统计
    print_statistics(stats, len(df))
    
    # 保存
    df.to_csv(RAW_DATA_PATH, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 已保存到: {RAW_DATA_PATH}")
    
    # 保存已标注数据
    labeled_df = df[df[TARGET] != -1]
    if len(labeled_df) > 0:
        labeled_df.to_csv(LABELED_DATA_PATH, index=False, encoding='utf-8-sig')
        print(f"[OK] 已保存 {len(labeled_df)} 条已标注数据到: {LABELED_DATA_PATH}")
    
    # 下一步提示
    print("\n" + "=" * 80)
    print("标注完成！")
    print("=" * 80)
    print(f"\n下一步:")
    print(f"  1. 批量训练所有模型: python train_all_models.py")
    print(f"  2. 单独训练某个模型: python models/lightgbm/train.py")
    print(f"  3. 生成测试集: python utils/create_test_set.py")
    print(f"  4. 模型对比测试: python models/comparison/test_all_models.py")


if __name__ == '__main__':
    main()
