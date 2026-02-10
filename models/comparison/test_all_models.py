"""
模型对比测试脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
from matplotlib import rcParams
from config import *
from utils.create_test_set import create_test_set

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 结果目录
RESULTS_DIR = f'{MODEL_DIR}/comparison/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 模型配置
MODELS = {
    'LightGBM': {
        'path': f'{MODEL_DIR}/lightgbm/model.txt',
        'type': 'lightgbm',
        'scaler_path': None
    },
    'Logistic': {
        'path': f'{MODEL_DIR}/logistic/model.pkl',
        'type': 'sklearn',
        'scaler_path': f'{MODEL_DIR}/logistic/scaler.pkl'
    },
    'SVM': {
        'path': f'{MODEL_DIR}/svm/model.pkl',
        'type': 'sklearn',
        'scaler_path': f'{MODEL_DIR}/svm/scaler.pkl'
    },
    'KNN': {
        'path': f'{MODEL_DIR}/knn/model.pkl',
        'type': 'sklearn',
        'scaler_path': f'{MODEL_DIR}/knn/scaler.pkl'
    },
    'RandomForest': {
        'path': f'{MODEL_DIR}/random_forest/model.pkl',
        'type': 'sklearn',
        'scaler_path': None
    },
    'MLP': {
        'path': f'{MODEL_DIR}/mlp/model.pkl',
        'type': 'sklearn',
        'scaler_path': f'{MODEL_DIR}/mlp/scaler.pkl'
    }
}


def load_or_create_test_set():
    """加载或创建测试集"""
    print("=" * 80)
    print("1. 准备测试数据")
    print("=" * 80)
    
    # 尝试加载已有的测试数据
    test_files = []
    if os.path.exists('data/test_sets'):
        test_files = [f for f in os.listdir('data/test_sets') if f.endswith('.csv')]
    
    if test_files:
        print(f"\n发现 {len(test_files)} 个测试集文件")
        print("使用最新的测试集...")
        test_files.sort(reverse=True)
        test_path = f'data/test_sets/{test_files[0]}'
        df = pd.read_csv(test_path, encoding='utf-8-sig')
        print(f"[OK] 加载测试集: {test_path}")
        print(f"样本数量: {len(df)}")
    else:
        print("\n未找到测试集，生成新的测试数据...")
        df, _ = create_test_set(n_samples=100, random_seed=42)
        print("[OK] 测试数据生成完成")
    
    # 只保留已标注的数据
    df = df[df[TARGET] != -1]
    print(f"已标注样本: {len(df)} 条")
    
    X_test = df[ALL_FEATURES]
    y_test = df[TARGET]
    
    return X_test, y_test


def load_model(model_name, model_config):
    """加载模型"""
    if not os.path.exists(model_config['path']):
        print(f"警告: {model_name} 模型文件不存在: {model_config['path']}")
        return None, None
    
    # 加载模型
    if model_config['type'] == 'lightgbm':
        model = lgb.Booster(model_file=model_config['path'])
    else:
        with open(model_config['path'], 'rb') as f:
            model = pickle.load(f)
    
    # 加载标准化器（如果有）
    scaler = None
    if model_config['scaler_path'] and os.path.exists(model_config['scaler_path']):
        with open(model_config['scaler_path'], 'rb') as f:
            scaler = pickle.load(f)
    
    return model, scaler


def predict_with_model(model, scaler, X_test, model_type):
    """使用模型进行预测"""
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    if model_type == 'lightgbm':
        y_pred_proba = model.predict(X_test_scaled)
    else:
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return y_pred_proba


def test_all_models(X_test, y_test):
    """测试所有模型"""
    print("\n" + "=" * 80)
    print("2. 测试所有模型")
    print("=" * 80)
    
    results = {}
    
    for model_name, model_config in MODELS.items():
        print(f"\n--- 测试 {model_name} ---")
        
        model, scaler = load_model(model_name, model_config)
        if model is None:
            print(f"跳过 {model_name}")
            continue
        
        # 预测
        y_pred_proba = predict_with_model(model, scaler, X_test, model_config['type'])
        y_pred = (y_pred_proba >= AUTO_MERGE_THRESHOLD).astype(int)
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1:     {f1:.4f}")
        print(f"AUC:    {auc:.4f}")
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'cm': cm
        }
    
    return results


def print_comparison_table(results):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("3. 模型对比")
    print("=" * 80)
    
    print(f"\n{'模型':<15} {'准确率':>8} {'精确率':>8} {'召回率':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 65)
    
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['accuracy']:>8.4f} {result['precision']:>8.4f} "
              f"{result['recall']:>8.4f} {result['f1']:>8.4f} {result['auc']:>8.4f}")


def visualize_comparison(results, y_test):
    """生成对比图表"""
    print("\n" + "=" * 80)
    print("4. 生成对比图表")
    print("=" * 80)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    model_names = list(results.keys())
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    
    # 1. 准确率对比
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [results[m]['accuracy'] for m in model_names]
    bars = ax1.bar(range(len(model_names)), accuracies, color=colors[:len(model_names)], edgecolor='black')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('准确率', fontsize=10)
    ax1.set_title('准确率对比', fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 2. AUC对比
    ax2 = fig.add_subplot(gs[0, 1])
    aucs = [results[m]['auc'] for m in model_names]
    bars = ax2.bar(range(len(model_names)), aucs, color=colors[:len(model_names)], edgecolor='black')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('AUC', fontsize=10)
    ax2.set_title('AUC对比', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 3. ROC曲线对比
    ax3 = fig.add_subplot(gs[0, 2])
    for i, model_name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_pred_proba'])
        auc_val = results[model_name]['auc']
        ax3.plot(fpr, tpr, label=f'{model_name} (AUC={auc_val:.3f})', 
                color=colors[i], linewidth=2)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax3.set_xlabel('假阳性率', fontsize=10)
    ax3.set_ylabel('真阳性率', fontsize=10)
    ax3.set_title('ROC曲线对比', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8, loc='lower right')
    ax3.grid(alpha=0.3)
    
    # 4. 精确率对比
    ax4 = fig.add_subplot(gs[1, 0])
    precisions = [results[m]['precision'] for m in model_names]
    bars = ax4.bar(range(len(model_names)), precisions, color=colors[:len(model_names)], edgecolor='black')
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('精确率', fontsize=10)
    ax4.set_title('精确率对比', fontsize=11, fontweight='bold')
    ax4.set_ylim([0, 1.05])
    ax4.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, precisions):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 5. 召回率对比
    ax5 = fig.add_subplot(gs[1, 1])
    recalls = [results[m]['recall'] for m in model_names]
    bars = ax5.bar(range(len(model_names)), recalls, color=colors[:len(model_names)], edgecolor='black')
    ax5.set_xticks(range(len(model_names)))
    ax5.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('召回率', fontsize=10)
    ax5.set_title('召回率对比', fontsize=11, fontweight='bold')
    ax5.set_ylim([0, 1.05])
    ax5.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, recalls):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 6. F1对比
    ax6 = fig.add_subplot(gs[1, 2])
    f1s = [results[m]['f1'] for m in model_names]
    bars = ax6.bar(range(len(model_names)), f1s, color=colors[:len(model_names)], edgecolor='black')
    ax6.set_xticks(range(len(model_names)))
    ax6.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax6.set_ylabel('F1 分数', fontsize=10)
    ax6.set_title('F1对比', fontsize=11, fontweight='bold')
    ax6.set_ylim([0, 1.05])
    ax6.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax6.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, f1s):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 7-9. 混淆矩阵（显示前3个模型）
    for idx, model_name in enumerate(model_names[:3]):
        ax = fig.add_subplot(gs[2, idx])
        cm = results[model_name]['cm']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ax.imshow(cm, cmap='Blues', alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['预测0', '预测1'], fontsize=9)
        ax.set_yticklabels(['实际0', '实际1'], fontsize=9)
        ax.set_title(f'{model_name} 混淆矩阵', fontsize=10, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)',
                       ha="center", va="center", color=text_color, fontsize=10)
    
    output_path = f'{RESULTS_DIR}/comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图表已保存到: {output_path}")
    plt.close()


def visualize_individual_model(model_name, result, y_test):
    """生成单个模型的测试结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 混淆矩阵
    cm = result['cm']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    axes[0, 0].imshow(cm, cmap='Blues', alpha=0.8)
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_xticklabels(['预测不合并', '预测合并'], fontsize=10)
    axes[0, 0].set_yticklabels(['实际不合并', '实际合并'], fontsize=10)
    axes[0, 0].set_title('混淆矩阵', fontsize=12, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[0, 0].text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)',
                           ha="center", va="center", color=text_color, 
                           fontsize=14, fontweight='bold')
    
    # 2. 预测概率分布
    y_pred_proba = result['y_pred_proba']
    proba_negative = y_pred_proba[y_test == 0]
    proba_positive = y_pred_proba[y_test == 1]
    
    bins = np.linspace(0, 1, 21)
    if len(proba_negative) > 0:
        axes[0, 1].hist(proba_negative, bins=bins, alpha=0.6, 
                       label=f'不可以合并 (n={len(proba_negative)})', 
                       color='red', edgecolor='darkred')
    if len(proba_positive) > 0:
        axes[0, 1].hist(proba_positive, bins=bins, alpha=0.6, 
                       label=f'可以合并 (n={len(proba_positive)})', 
                       color='green', edgecolor='darkgreen')
    
    axes[0, 1].axvline(x=AUTO_MERGE_THRESHOLD, color='black', linestyle='--', 
                      linewidth=2, label=f'阈值={AUTO_MERGE_THRESHOLD}')
    axes[0, 1].set_xlabel('预测概率', fontsize=10)
    axes[0, 1].set_ylabel('样本数', fontsize=10)
    axes[0, 1].set_title('预测概率分布', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='upper center', fontsize=9)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 评估指标
    metrics = ['准确率', '精确率', '召回率', 'F1', 'AUC']
    values = [
        result['accuracy'],
        result['precision'],
        result['recall'],
        result['f1'],
        result['auc']
    ]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = axes[1, 0].bar(metrics, values, color=colors, edgecolor='black', alpha=0.8)
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].set_ylabel('分数', fontsize=10)
    axes[1, 0].set_title('模型评估指标', fontsize=12, fontweight='bold')
    axes[1, 0].axhline(y=0.8, color='gray', linestyle='--', alpha=0.7)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for v, bar in zip(values, bars):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. 预测结果统计
    axes[1, 1].axis('off')
    stats_text = f"""测试结果统计

测试样本数: {len(y_test)}

预测为可合并: {(result['y_pred'] == 1).sum()}
预测为不可合并: {(result['y_pred'] == 0).sum()}

实际可合并: {(y_test == 1).sum()}
实际不可合并: {(y_test == 0).sum()}

正确预测数: {(result['y_pred'] == y_test).sum()}
错误预测数: {(result['y_pred'] != y_test).sum()}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                   verticalalignment='center', family='monospace')
    axes[1, 1].set_title('测试结果统计', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = f'{RESULTS_DIR}/{model_name.lower()}_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  → {model_name} 测试结果已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    try:
        # 1. 加载测试数据
        X_test, y_test = load_or_create_test_set()
        
        # 2. 测试所有模型
        results = test_all_models(X_test, y_test)
        
        if not results:
            print("\n错误: 没有可用的模型")
            return
        
        # 3. 打印对比表格
        print_comparison_table(results)
        
        # 4. 生成单个模型测试结果
        print("\n" + "=" * 80)
        print("5. 生成单个模型测试结果")
        print("=" * 80)
        for model_name, result in results.items():
            visualize_individual_model(model_name, result, y_test)
        
        # 5. 生成对比图表
        visualize_comparison(results, y_test)
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        print(f"\n结果目录: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
