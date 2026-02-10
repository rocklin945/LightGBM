"""
Random Forest 模型训练脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
from matplotlib import rcParams
from config import *

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

MODEL_NAME = 'random_forest'
MODEL_PATH = f'{MODEL_DIR}/{MODEL_NAME}/model.pkl'
RESULTS_DIR = f'{MODEL_DIR}/{MODEL_NAME}/results'

os.makedirs(f'{MODEL_DIR}/{MODEL_NAME}', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_split_data():
    """加载和划分数据"""
    print("=" * 80)
    print("1. 加载数据 - Random Forest")
    print("=" * 80)
    
    df = pd.read_csv(LABELED_DATA_PATH, encoding='utf-8-sig')
    print(f"\n已加载 {len(df)} 条标注数据")
    
    X = df[ALL_FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2024, stratify=y
    )
    
    print(f"\n训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """训练模型"""
    print("\n" + "=" * 80)
    print("2. 训练 Random Forest 模型")
    print("=" * 80)
    
    print("\n开始训练...")
    model = RandomForestClassifier(n_estimators=100, random_state=2024)
    model.fit(X_train, y_train)
    
    print("训练完成！")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"模型已保存到: {MODEL_PATH}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """评估模型"""
    print("\n" + "=" * 80)
    print("3. 模型评估")
    print("=" * 80)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= AUTO_MERGE_THRESHOLD).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n评估指标（阈值={AUTO_MERGE_THRESHOLD}）:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1:     {f1:.4f}")
    print(f"  AUC:    {auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'cm': cm
    }


def visualize_results(model, X_test, y_test, eval_results):
    """可视化结果"""
    print("\n" + "=" * 80)
    print("4. 生成可视化图表")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 特征重要性
    importance = model.feature_importances_
    feature_names = ALL_FEATURES
    
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'description': [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names]
    }).sort_values('importance', ascending=False).head(15)
    
    # 保存特征重要性
    feature_df.to_csv(f'{RESULTS_DIR}/feature_importance.csv', index=False, encoding='utf-8-sig')
    
    y_pos = np.arange(len(feature_df))
    axes[0, 0].barh(y_pos, feature_df['importance'].values, color='steelblue', edgecolor='black')
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(feature_df['description'].values, fontsize=8)
    axes[0, 0].set_xlabel('重要性', fontsize=10)
    axes[0, 0].set_title('Top 15 特征重要性', fontsize=12, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. 预测概率分布
    y_pred_proba = eval_results['y_pred_proba']
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
    
    # 3. 混淆矩阵
    cm = eval_results['cm']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    axes[1, 0].imshow(cm, cmap='Blues', alpha=0.8)
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['预测不合并', '预测合并'], fontsize=10)
    axes[1, 0].set_yticklabels(['实际不合并', '实际合并'], fontsize=10)
    axes[1, 0].set_title('混淆矩阵', fontsize=12, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[1, 0].text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)',
                           ha="center", va="center", color=text_color, 
                           fontsize=14, fontweight='bold')
    
    # 4. 评估指标
    metrics = ['准确率', '精确率', '召回率', 'F1', 'AUC']
    values = [
        eval_results['accuracy'],
        eval_results['precision'],
        eval_results['recall'],
        eval_results['f1'],
        eval_results['auc']
    ]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = axes[1, 1].bar(metrics, values, color=colors, edgecolor='black', alpha=0.8)
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].set_ylabel('分数', fontsize=10)
    axes[1, 1].set_title('模型评估指标', fontsize=12, fontweight='bold')
    axes[1, 1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.7)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for v, bar in zip(values, bars):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = f'{RESULTS_DIR}/training_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化图表已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    try:
        X_train, X_test, y_train, y_test = load_and_split_data()
        model = train_model(X_train, y_train)
        eval_results = evaluate_model(model, X_test, y_test)
        visualize_results(model, X_test, y_test, eval_results)
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
