"""
测试模型：生成测试数据 -> 自动标注 -> 模型预测 -> 评估（包含混淆矩阵）
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from config import *

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def generate_test_data(n_samples=200):
    """生成测试数据（和训练数据相同的生成逻辑）"""
    print(f"生成 {n_samples} 条测试数据...")
    
    np.random.seed(None)  # 使用随机种子，每次生成不同数据
    
    data = {
        'mr_id': [f'TEST-{i+1:05d}' for i in range(n_samples)],
        'mr_title': [f'Test MR #{i+1}' for i in range(n_samples)],
        'author': [f'author_{np.random.randint(1, 30)}' for _ in range(n_samples)],
        'repo': [f'repo_{np.random.randint(1, 15)}' for _ in range(n_samples)],
    }
    
    # 生成特征值（与 1_data_labeling.py 相同的逻辑）
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
    
    df = pd.DataFrame(data)
    print(f"[OK] 生成完成: {len(df)} 条")
    return df


def auto_label_test_data(df):
    """使用 auto_label.py 的规则自动标注测试数据（获得真实标签）"""
    print("\n使用规则自动标注测试数据（作为真实标签）...")
    
    # 初始化标签为 -1
    df[TARGET] = -1
    
    for idx, row in df.iterrows():
        label = -1
        
        # 规则1: 安全可合并
        if row['has_only_doc_change'] == 1:
            label = 1
        elif row['has_only_test_change'] == 1 and row['ci_unit_test_passed'] == 1:
            label = 1
        elif (row['is_repo_owner'] == 1 or row['is_maintainer'] == 1):
            if (row['total_changed_lines'] <= 30 and 
                row['changed_files_count'] <= 2 and
                row['ci_build_passed'] == 1 and
                row['ci_unit_test_passed'] == 1):
                label = 1
        elif row['is_core_contributor'] == 1:
            if (row['total_changed_lines'] <= 50 and
                row['changed_files_count'] <= 3 and
                row['ci_build_passed'] == 1 and
                row['ci_unit_test_passed'] == 1 and
                row['ci_lint_passed'] == 1):
                label = 1
        elif (row['author_total_mrs'] >= 50 and
              row['author_merge_rate_30d'] >= 0.85 and
              row['author_revert_rate_30d'] <= 0.03):
            if (row['total_changed_lines'] <= 100 and
                row['changed_files_count'] <= 5 and
                row['ci_build_passed'] == 1 and
                row['has_unit_test'] == 1 and
                row['unit_test_coverage'] >= 70):
                label = 1
        
        # 规则2: 必须拒绝
        if label == -1:
            if row['involves_payment'] == 1:
                label = 0
            elif row['ci_build_passed'] == 0:
                label = 0
            elif row['involves_pii'] == 1:
                label = 0
            elif row['has_breaking_change'] == 1:
                label = 0
            elif row['has_database_migration'] == 1:
                label = 0
            elif row['security_vulnerabilities'] > 0:
                label = 0
            elif row['total_changed_lines'] > 500:
                label = 0
            elif row['changed_files_count'] > 15:
                label = 0
            elif row['is_first_mr'] == 1 and row['total_changed_lines'] > 50:
                label = 0
            elif row['author_total_mrs'] < 3:
                label = 0
            elif row['ci_unit_test_passed'] == 0:
                label = 0
            elif row['code_lines_ratio'] > 0.7 and row['has_unit_test'] == 0:
                label = 0
            elif row['test_coverage_change'] < -15:
                label = 0
            elif row['has_todo_or_fixme'] == 1:
                label = 0
            elif row['lint_issues_count'] > 15:
                label = 0
            elif row['code_smells_count'] > 15:
                label = 0
        
        # 规则3: 中等风险倾向拒绝
        if label == -1:
            if row['involves_auth'] == 1 or row['involves_security'] == 1:
                label = 0
            elif row['modifies_critical_file'] == 1:
                label = 0
            elif row['has_api_change'] == 1 and row['is_maintainer'] == 0:
                label = 0
            elif row['total_changed_lines'] > 200 and row['unit_test_coverage'] < 60:
                label = 0
            elif row['has_new_dependency'] == 1 and row['is_core_contributor'] == 0:
                label = 0
        
        # 规则4: 中等风险倾向通过
        if label == -1:
            if (row['author_total_mrs'] >= 10 and
                row['author_merge_rate_30d'] >= 0.75):
                if (row['total_changed_lines'] <= 300 and
                    row['changed_files_count'] <= 8 and
                    row['ci_build_passed'] == 1 and
                    row['ci_unit_test_passed'] == 1 and
                    row['ci_lint_passed'] == 1):
                    label = 1
        
        df.at[idx, TARGET] = label
    
    # 统计标注结果
    n_can = len(df[df[TARGET] == 1])
    n_cannot = len(df[df[TARGET] == 0])
    n_uncertain = len(df[df[TARGET] == -1])
    
    print(f"[OK] 自动标注完成:")
    print(f"  - 可以合并:   {n_can:4d} ({n_can/len(df)*100:5.1f}%)")
    print(f"  - 不可合并:   {n_cannot:4d} ({n_cannot/len(df)*100:5.1f}%)")
    print(f"  - 不确定:     {n_uncertain:4d} ({n_uncertain/len(df)*100:5.1f}%)")
    
    # 只保留已标注的数据
    labeled_df = df[df[TARGET] != -1].copy()
    print(f"\n保留 {len(labeled_df)} 条已标注数据用于测试")
    
    return labeled_df


def test_model(df, model_path=MODEL_PATH):
    """使用训练好的模型预测测试数据"""
    print("\n加载模型并预测...")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先运行: python 2_train_model.py 训练模型")
        return None
    
    # 加载模型
    model = lgb.Booster(model_file=model_path)
    print(f"[OK] 模型加载成功")
    
    # 准备特征和标签
    X_test = df[ALL_FEATURES]
    y_test = df[TARGET].values
    
    # 预测
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= AUTO_MERGE_THRESHOLD).astype(int)
    
    print(f"[OK] 预测完成")
    
    return model, y_test, y_pred, y_pred_proba


def evaluate_model(y_test, y_pred, y_pred_proba):
    """评估模型性能"""
    print("\n" + "=" * 80)
    print("模型评估")
    print("=" * 80)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n评估指标（阈值={AUTO_MERGE_THRESHOLD}）:")
    print(f"  准确率 (Accuracy):  {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1 分数:            {f1:.4f}")
    print(f"  AUC:                {auc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(f"                预测不合并  预测合并")
    print(f"  实际不合并      {cm[0,0]:6d}     {cm[0,1]:6d}")
    print(f"  实际合并        {cm[1,0]:6d}     {cm[1,1]:6d}")
    
    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(
        y_test, y_pred, 
        target_names=['不可以自动合并', '可以自动合并'],
        zero_division=0
    ))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def visualize_test_results(model, y_test, y_pred, y_pred_proba, eval_results, 
                           output_path='d:/LightGBM/models/test_results.png'):
    """生成测试结果可视化（和训练结果相同的格式）"""
    print("\n生成可视化图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # === 1. 特征重要性（Top 15）===
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    
    # 获取Top 15特征
    indices = np.argsort(importance)[::-1][:15]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]
    
    # 转换特征名为中文
    top_features_cn = [FEATURE_DESCRIPTIONS.get(f, f) for f in top_features]
    
    # 绘制水平条形图
    y_pos = np.arange(len(top_features_cn))
    axes[0, 0].barh(y_pos, top_importance, color='steelblue', edgecolor='black')
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(top_features_cn, fontsize=8)
    axes[0, 0].set_xlabel('重要性 (gain)', fontsize=10)
    axes[0, 0].set_title('Top 15 特征重要性', fontsize=12, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # === 2. 预测概率分布 ===
    proba_negative = y_pred_proba[y_test == 0]
    proba_positive = y_pred_proba[y_test == 1]
    
    bins = np.linspace(0, 1, 21)
    
    if len(proba_negative) > 0:
        axes[0, 1].hist(proba_negative, bins=bins, alpha=0.6, 
                       label=f'不可以合并 (n={len(proba_negative)})', 
                       color='red', edgecolor='darkred', linewidth=1)
    
    if len(proba_positive) > 0:
        axes[0, 1].hist(proba_positive, bins=bins, alpha=0.6, 
                       label=f'可以合并 (n={len(proba_positive)})', 
                       color='green', edgecolor='darkgreen', linewidth=1)
    
    axes[0, 1].axvline(x=AUTO_MERGE_THRESHOLD, color='black', linestyle='--', 
                      linewidth=2, label=f'阈值={AUTO_MERGE_THRESHOLD}')
    axes[0, 1].set_xlabel('预测概率', fontsize=10)
    axes[0, 1].set_ylabel('样本数', fontsize=10)
    axes[0, 1].set_title('预测概率分布', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='upper center', fontsize=9)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # === 3. 混淆矩阵热力图 ===
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = axes[1, 0].imshow(cm, cmap='Blues', alpha=0.8)
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['预测不合并', '预测合并'], fontsize=10)
    axes[1, 0].set_yticklabels(['实际不合并', '实际合并'], fontsize=10)
    axes[1, 0].set_title('混淆矩阵', fontsize=12, fontweight='bold')
    
    # 在格子中显示数字和百分比
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[1, 0].text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)',
                           ha="center", va="center", color=text_color, 
                           fontsize=14, fontweight='bold')
    
    # === 4. 评估指标对比 ===
    metrics = ['准确率', '精确率', '召回率', 'F1', 'AUC']
    values = [
        eval_results['accuracy'],
        eval_results['precision'],
        eval_results['recall'],
        eval_results['f1'],
        eval_results['auc']
    ]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = axes[1, 1].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].set_ylabel('分数', fontsize=10)
    axes[1, 1].set_title('模型评估指标', fontsize=12, fontweight='bold')
    axes[1, 1].axhline(y=0.8, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='良好阈值(0.8)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].legend(fontsize=8)
    
    # 在柱子上显示数值
    for i, (v, bar) in enumerate(zip(values, bars)):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 可视化已保存: {output_path}")
    print("\n正在显示图表窗口...")
    plt.show()


def main():
    """主函数"""
    print("=" * 80)
    print("模型测试程序")
    print("=" * 80)
    
    # 询问生成多少测试数据
    try:
        n_test = input("\n生成多少条测试数据？(默认 200): ").strip()
        n_test = int(n_test) if n_test else 200
    except:
        n_test = 200
    
    # 1. 生成测试数据
    test_df = generate_test_data(n_samples=n_test)
    
    # 2. 自动标注测试数据（获得真实标签）
    labeled_df = auto_label_test_data(test_df)
    
    if len(labeled_df) == 0:
        print("\n错误: 没有可用的标注数据")
        return
    
    # 3. 使用模型预测
    result = test_model(labeled_df)
    if result is None:
        return
    
    model, y_test, y_pred, y_pred_proba = result
    
    # 4. 评估模型
    eval_results = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # 5. 生成可视化
    visualize_test_results(model, y_test, y_pred, y_pred_proba, eval_results)
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print("\n测试结果已保存:")
    print("  可视化: d:/LightGBM/models/test_results.png")
    print("\n包含:")
    print("  1. 特征重要性 (Top 15)")
    print("  2. 预测概率分布")
    print("  3. 混淆矩阵（重点）")
    print("  4. 评估指标对比")


if __name__ == '__main__':
    main()
