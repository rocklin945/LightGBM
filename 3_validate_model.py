"""
MR 风险评估模型验证程序
用于验证模型在新数据上的表现
"""

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


class MRRiskModelValidator:
    def __init__(self):
        """初始化验证器"""
        self.model = None
    
    def load_model(self):
        """加载训练好的模型"""
        print("=" * 80)
        print("MR 风险评估模型验证")
        print("=" * 80)
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"未找到模型文件: {MODEL_PATH}\n"
                f"请先运行 2_train_model.py 训练模型"
            )
        
        print(f"\n加载模型: {MODEL_PATH}")
        self.model = lgb.Booster(model_file=MODEL_PATH)
        print("模型加载成功！")
    
    def predict_single_mr(self, mr_data):
        """
        预测单个 MR 是否可以自动合并
        
        参数:
            mr_data: dict，包含所有特征的字典
        
        返回:
            dict: 包含预测结果和详细信息
        """
        # 构建特征向量
        features = [mr_data.get(f, 0) for f in ALL_FEATURES]
        features_array = np.array(features).reshape(1, -1)
        
        # 预测概率
        prob = self.model.predict(features_array)[0]
        
        # 判断是否可以自动合并
        can_merge = prob >= AUTO_MERGE_THRESHOLD
        
        # 检查强制拒绝条件
        force_reject_reasons = []
        for condition, expected_value in FORCE_REJECT_CONDITIONS.items():
            if mr_data.get(condition, 0) == expected_value:
                force_reject_reasons.append(FEATURE_DESCRIPTIONS[condition])
        
        if force_reject_reasons:
            can_merge = False
            final_decision = "强制拒绝"
        else:
            final_decision = "可以自动合并" if can_merge else "需要人工审核"
        
        return {
            'probability': prob,
            'can_auto_merge': can_merge and len(force_reject_reasons) == 0,
            'decision': final_decision,
            'force_reject_reasons': force_reject_reasons,
            'confidence': 'high' if abs(prob - 0.5) > 0.3 else 'low'
        }
    
    def validate_test_set(self):
        """在测试集上验证模型"""
        print("\n" + "=" * 80)
        print("1. 测试集验证")
        print("=" * 80)
        
        if not os.path.exists(TEST_DATA_PATH):
            print(f"\n未找到测试集文件: {TEST_DATA_PATH}")
            print("请先运行 2_train_model.py 生成测试集")
            return None
        
        # 加载测试集
        test_df = pd.read_csv(TEST_DATA_PATH, encoding='utf-8-sig')
        print(f"\n加载测试集: {len(test_df)} 条数据")
        
        X_test = test_df[ALL_FEATURES]
        y_test = test_df[TARGET]
        
        # 预测
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= AUTO_MERGE_THRESHOLD).astype(int)
        
        # 应用强制拒绝规则
        for idx, row in test_df.iterrows():
            for condition, expected_value in FORCE_REJECT_CONDITIONS.items():
                if row[condition] == expected_value:
                    y_pred[idx] = 0  # 强制拒绝
                    break
        
        # 评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n评估指标:")
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
        
        # 错误分析
        print("\n" + "=" * 80)
        print("2. 错误分析")
        print("=" * 80)
        
        # 假阳性（预测可以合并，实际不可以）
        false_positives = test_df[(y_pred == 1) & (y_test == 0)]
        print(f"\n假阳性（高风险误判）: {len(false_positives)} 条")
        if len(false_positives) > 0:
            print("详细信息（前3条）:")
            for idx, row in false_positives.head(3).iterrows():
                print(f"  - MR ID: {row.get('mr_id', 'N/A')}, 概率: {y_pred_proba[idx]:.3f}")
        
        # 假阴性（预测不可以合并，实际可以）
        false_negatives = test_df[(y_pred == 0) & (y_test == 1)]
        print(f"\n假阴性（保守误判）: {len(false_negatives)} 条")
        if len(false_negatives) > 0:
            print("详细信息（前3条）:")
            for idx, row in false_negatives.head(3).iterrows():
                print(f"  - MR ID: {row.get('mr_id', 'N/A')}, 概率: {y_pred_proba[idx]:.3f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives)
        }
    
    def demo_prediction(self):
        """演示预测功能"""
        print("\n" + "=" * 80)
        print("3. 预测演示")
        print("=" * 80)
        
        # 示例 1: 低风险 MR
        print("\n【示例 1: 低风险 MR】")
        low_risk_mr = {
            'changed_lines': 10,
            'affected_methods': 2,
            'file_type': 2,  # md 文件
            'change_type': 1,  # 注释
            'method_count_change': 0,
            'line_count_change': 10,
            'class_count_change': 0,
            'nesting_level_change': 0,
            'is_core_repo': 0,
            'repo_importance_score': 3,
            'ci_passed': 1,
            'test_coverage_change': 0,
            'has_unit_test': 1,
            'recent_mr_count': 20,
            'mr_to_lgtm_interval': 2,
            'author_mr_ratio': 0.8,
            'avg_review_time': 3,
            'is_repo_owner': 1,
            'is_maintainer': 1,
            'author_experience_days': 500,
            'author_total_mr_count': 100,
            'ai_cr_passed': 1,
            'ai_cr_issue_count': 0,
            'invalid_ai_cr_count': 0,
            'involves_auth': 0,
            'involves_payment': 0,
            'involves_security': 0,
            'involves_encryption': 0,
        }
        
        result = self.predict_single_mr(low_risk_mr)
        print(f"预测概率: {result['probability']:.4f}")
        print(f"决策: {result['decision']}")
        print(f"置信度: {result['confidence']}")
        
        # 示例 2: 高风险 MR
        print("\n【示例 2: 高风险 MR】")
        high_risk_mr = {
            'changed_lines': 500,
            'affected_methods': 50,
            'file_type': 4,  # 代码文件
            'change_type': 4,  # 代码改动
            'method_count_change': 20,
            'line_count_change': 500,
            'class_count_change': 5,
            'nesting_level_change': 3,
            'is_core_repo': 1,
            'repo_importance_score': 9,
            'ci_passed': 0,  # CI 未通过
            'test_coverage_change': -10,
            'has_unit_test': 0,
            'recent_mr_count': 2,
            'mr_to_lgtm_interval': 48,
            'author_mr_ratio': 0.1,
            'avg_review_time': 24,
            'is_repo_owner': 0,
            'is_maintainer': 0,
            'author_experience_days': 30,
            'author_total_mr_count': 5,
            'ai_cr_passed': 0,
            'ai_cr_issue_count': 15,
            'invalid_ai_cr_count': 0,
            'involves_auth': 1,  # 涉及权限
            'involves_payment': 0,
            'involves_security': 0,
            'involves_encryption': 0,
        }
        
        result = self.predict_single_mr(high_risk_mr)
        print(f"预测概率: {result['probability']:.4f}")
        print(f"决策: {result['decision']}")
        print(f"置信度: {result['confidence']}")
        if result['force_reject_reasons']:
            print(f"强制拒绝原因: {', '.join(result['force_reject_reasons'])}")
    
    def run(self):
        """运行验证流程"""
        try:
            # 加载模型
            self.load_model()
            
            # 验证测试集
            self.validate_test_set()
            
            # 演示预测
            self.demo_prediction()
            
            print("\n" + "=" * 80)
            print("验证完成！")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    validator = MRRiskModelValidator()
    validator.run()


if __name__ == '__main__':
    main()
