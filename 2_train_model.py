"""
MR 风险评估模型训练程序
使用 LightGBM 训练二分类模型
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
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


class MRRiskModelTrainer:
    def __init__(self):
        """初始化训练器"""
        self.model = None
        self.train_data = None
        self.test_data = None
        self.feature_importance = None
    
    def load_data(self):
        """加载标注数据"""
        print("=" * 80)
        print("1. 加载数据")
        print("=" * 80)
        
        if not os.path.exists(LABELED_DATA_PATH):
            raise FileNotFoundError(
                f"未找到标注数据文件: {LABELED_DATA_PATH}\n"
                f"请先运行 1_data_labeling.py 进行数据标注"
            )
        
        df = pd.read_csv(LABELED_DATA_PATH, encoding='utf-8-sig')
        print(f"\n已加载 {len(df)} 条标注数据")
        
        # 检查数据质量
        n_positive = len(df[df[TARGET] == 1])
        n_negative = len(df[df[TARGET] == 0])
        
        print(f"  - 可以自动合并: {n_positive} ({n_positive/len(df)*100:.1f}%)")
        print(f"  - 不可以自动合并: {n_negative} ({n_negative/len(df)*100:.1f}%)")
        
        if n_positive < 5 or n_negative < 5:
            print("\n警告：正负样本数量不足，建议至少各有 10 条以上数据")
        
        return df
    
    def split_data(self, df, test_size=0.2):
        """划分训练集和测试集"""
        print("\n" + "=" * 80)
        print("2. 划分训练集和测试集")
        print("=" * 80)
        
        X = df[ALL_FEATURES]
        y = df[TARGET]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=2024,
            stratify=y  # 保持正负样本比例
        )
        
        print(f"\n训练集: {len(X_train)} 条")
        print(f"测试集: {len(X_test)} 条")
        print(f"特征数量: {len(ALL_FEATURES)}")
        
        # 保存划分后的数据
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(TRAIN_DATA_PATH, index=False, encoding='utf-8-sig')
        test_df.to_csv(TEST_DATA_PATH, index=False, encoding='utf-8-sig')
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_test, y_test):
        """训练模型"""
        print("\n" + "=" * 80)
        print("3. 训练 LightGBM 模型")
        print("=" * 80)
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 训练模型
        print("\n开始训练...")
        self.model = lgb.train(
            LIGHTGBM_PARAMS,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        print(f"\n训练完成！最佳迭代次数: {self.model.best_iteration}")
        
        # 保存模型
        self.model.save_model(MODEL_PATH)
        print(f"模型已保存到: {MODEL_PATH}")
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        print("\n" + "=" * 80)
        print("4. 模型评估")
        print("=" * 80)
        
        # 预测
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba >= AUTO_MERGE_THRESHOLD).astype(int)
        
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
            'auc': auc,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n" + "=" * 80)
        print("5. 特征重要性分析")
        print("=" * 80)
        
        # 获取特征重要性（使用 gain 类型，表示特征对模型的贡献）
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()
        
        print(f"\n特征数量: {len(feature_names)}")
        print(f"重要性值范围: {importance.min():.2f} - {importance.max():.2f}")
        print(f"重要性总和: {importance.sum():.2f}")
        
        # 创建 DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'description': [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names]
        }).sort_values('importance', ascending=False)
        
        # 保存
        self.feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False, encoding='utf-8-sig')
        print(f"\n特征重要性已保存到: {FEATURE_IMPORTANCE_PATH}")
        
        # 显示 Top 15
        print("\nTop 15 重要特征:")
        top_15 = self.feature_importance.head(15)
        for idx, row in top_15.iterrows():
            print(f"  {row['description']:<50} {row['importance']:>10.2f}")
        
        # 统计零重要性特征
        zero_importance = self.feature_importance[self.feature_importance['importance'] == 0]
        if len(zero_importance) > 0:
            print(f"\n警告: {len(zero_importance)} 个特征的重要性为 0（模型未使用）")
    
    def visualize_results(self, X_test, y_test, eval_results):
        """可视化结果"""
        print("\n" + "=" * 80)
        print("6. 生成可视化图表")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 6.1 特征重要性（Top 15）
        top_features = self.feature_importance.head(15)
        
        # 调试信息
        print(f"\n特征重要性范围: {top_features['importance'].min():.2f} - {top_features['importance'].max():.2f}")
        
        if len(top_features) > 0 and top_features['importance'].sum() > 0:
            # 使用水平条形图，从上到下排列
            y_pos = np.arange(len(top_features))
            axes[0, 0].barh(y_pos, top_features['importance'].values, color='steelblue', edgecolor='black')
            axes[0, 0].set_yticks(y_pos)
            axes[0, 0].set_yticklabels(top_features['description'].values, fontsize=8)
            axes[0, 0].set_xlabel('重要性 (gain)', fontsize=10)
            axes[0, 0].set_title('Top 15 特征重要性', fontsize=12, fontweight='bold')
            axes[0, 0].invert_yaxis()  # 最重要的在最上面
            axes[0, 0].grid(axis='x', alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, '特征重要性数据不足', 
                           ha='center', va='center', fontsize=12)
            axes[0, 0].set_title('Top 15 特征重要性', fontsize=12, fontweight='bold')
        
        # 6.2 预测概率分布
        y_pred_proba = eval_results['y_pred_proba']
        
        # 分别获取两类样本的预测概率
        proba_negative = y_pred_proba[y_test == 0]  # 不可以合并
        proba_positive = y_pred_proba[y_test == 1]  # 可以合并
        
        print(f"不可以合并样本数: {len(proba_negative)}")
        print(f"可以合并样本数: {len(proba_positive)}")
        
        # 使用不同的 bins 范围，确保两类都能显示
        bins = np.linspace(0, 1, 21)
        
        if len(proba_negative) > 0:
            axes[0, 1].hist(proba_negative, bins=bins, alpha=0.6, label=f'不可以合并 (n={len(proba_negative)})', 
                           color='red', edgecolor='darkred', linewidth=1)
        
        if len(proba_positive) > 0:
            axes[0, 1].hist(proba_positive, bins=bins, alpha=0.6, label=f'可以合并 (n={len(proba_positive)})', 
                           color='green', edgecolor='darkgreen', linewidth=1)
        
        axes[0, 1].axvline(x=AUTO_MERGE_THRESHOLD, color='black', linestyle='--', 
                          linewidth=2, label=f'阈值={AUTO_MERGE_THRESHOLD}')
        axes[0, 1].set_xlabel('预测概率', fontsize=10)
        axes[0, 1].set_ylabel('样本数', fontsize=10)
        axes[0, 1].set_title('预测概率分布', fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='upper center', fontsize=9)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 6.3 混淆矩阵热力图
        y_pred = eval_results['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        # 归一化显示百分比
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
        
        # 6.4 评估指标对比
        metrics = ['准确率', '精确率', '召回率', 'F1', 'AUC']
        values = [
            eval_results['accuracy'],
            eval_results['precision'],
            eval_results['recall'],
            eval_results['f1'],
            eval_results['auc']
        ]
        
        # 使用不同颜色表示不同指标，根据值的高低调整颜色深度
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
        output_path = f'{MODEL_DIR}/training_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n可视化图表已保存到: {output_path}")
        plt.show()
    
    def run(self):
        """运行完整训练流程"""
        try:
            # 1. 加载数据
            df = self.load_data()
            
            # 2. 划分数据集
            X_train, X_test, y_train, y_test = self.split_data(df)
            
            # 3. 训练模型
            self.train(X_train, y_train, X_test, y_test)
            
            # 4. 评估模型
            eval_results = self.evaluate(X_test, y_test)
            
            # 5. 特征重要性分析
            self.analyze_feature_importance()
            
            # 6. 可视化结果
            self.visualize_results(X_test, y_test, eval_results)
            
            print("\n" + "=" * 80)
            print("训练完成！")
            print("=" * 80)
            print(f"\n模型文件: {MODEL_PATH}")
            print(f"特征重要性: {FEATURE_IMPORTANCE_PATH}")
            print(f"可视化结果: {MODEL_DIR}/training_results.png")
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    trainer = MRRiskModelTrainer()
    trainer.run()


if __name__ == '__main__':
    main()
