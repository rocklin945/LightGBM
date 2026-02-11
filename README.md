# MR 风险评估与自动合并模型

## 📖 项目简介

本项目旨在通过机器学习模型，自动评估 Merge Request (MR) 的风险，判断是否可以跳过人工评审直接自动合并。

项目包含 6 种模型的实现：
- LightGBM
- Logistic Regression
- SVM
- KNN
- Random Forest
- MLP (多层感知机)

## 📁 目录结构

```
LightGBM/
├── data/                         # 所有数据文件
│   ├── raw_mr_data.csv          # 原始未标注数据
│   ├── labeled_data.csv         # 已标注数据（训练用）
│   ├── train_data.csv           # 训练集
│   ├── test_data.csv            # 验证集
│   └── test_sets/               # 测试集（可生成多个）
│
├── utils/                        # 工具脚本
│   ├── generate_data.py         # 统一数据生成（可控制数量）
│   ├── auto_label.py            # 规则自动标注
│   ├── llm_label.py             # LLM智能标注
│   └── create_test_set.py       # 生成测试集
│
├── models/                       # 所有模型
│   ├── lightgbm/
│   │   ├── train.py             # 训练脚本
│   │   ├── model.txt            # 模型文件
│   │   └── results/             # 结果文件夹
│   │       ├── training_results.png
│   │       └── feature_importance.csv
│   ├── logistic/
│   │   ├── train.py
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   └── results/
│   ├── svm/
│   │   ├── train.py
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   └── results/
│   ├── knn/
│   │   ├── train.py
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   └── results/
│   ├── random_forest/
│   │   ├── train.py
│   │   ├── model.pkl
│   │   └── results/
│   ├── mlp/
│   │   ├── train.py
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   └── results/
│   └── comparison/              # 模型对比
│       ├── train_all_models.py  # 批量训练所有模型
|       ├── test_all_models.py   # 对比测试脚本
│       └── results/             # 对比结果
│           ├── comparison.png
│           ├── lightgbm_test.png
│           ├── logistic_test.png
│           └── ...
│
├── config.py                     # 配置文件
├── 1_data_labeling.py           # 人工标注工具
└── README.md                     # 本文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib
```

### 2. 生成原始数据

```bash
python utils/generate_data.py
```

程序会询问生成数量，默认 500 条。

### 3. 自动标注（两种方式）

#### 方式A：规则标注（快速、免费）
```bash
python utils/auto_label.py
```

使用规则和评分系统全自动标注所有数据，无需人工介入：
- 规则标注：明确符合规则的数据
- 评分标注：对不确定的数据进行评分判断（60分及以上可合并）

#### 方式B：LLM标注（智能、需要API）
```bash
python utils/llm_label.py
```

使用大模型API进行智能标注：
- 更智能的判断逻辑
- 自动控制token消耗
- 批量处理，提供详细统计
- 适合对标注质量要求高的场景

### 4. 训练所有模型

```bash
python train_all_models.py
```

批量训练所有 6 个模型，每个模型会：
- 自动划分训练集和测试集
- 训练模型并保存
- 生成训练结果图表

### 6. 模型对比测试

```bash
python models/comparison/test_all_models.py
```

对所有模型进行测试对比，生成：
- 各模型单独的测试结果图表
- 综合对比图表（包含 ROC 曲线、混淆矩阵等）

## 📊 特征说明

模型使用 **50 个核心特征**，分为 10 个类别：

### 1. 代码变更规模（5个）
- 总改动行数
- 新增行数
- 删除行数
- 改动文件数
- 单个文件最大改动行数

### 2. 文件类型（5个）
- 代码文件数
- 测试文件数
- 是否仅改动测试
- 是否仅改动文档
- 代码改动占比

### 3. 代码复杂度（4个）
- 修改方法数
- 最大方法圈复杂度
- 最大方法行数
- 最大嵌套层级

### 4. 关键变更（4个）
- 是否新增依赖
- 是否有数据库迁移
- 是否有 API 变更
- 是否有破坏性变更

### 5. 测试质量（5个）
- 是否有单元测试
- 单元测试覆盖率
- 覆盖率变化
- 新增测试数量
- 测试代码比例

### 6. CI/CD 质量（6个）
- 构建是否通过
- 单元测试是否通过
- 代码规范检查是否通过
- 代码规范问题数
- 安全漏洞数
- 代码坏味道数

### 7. 作者质量（9个）
- 作者总 MR 数
- 作者最近 30 天 MR 数
- 作者最近 30 天合并率
- 作者最近 30 天回滚率
- 是否首个 MR
- 是否核心贡献者
- 作者在仓库天数
- 作者平均 MR 大小
- 作者 30 天 Bug 修复数

### 8. 提交质量（3个）
- 提交次数
- 平均提交大小
- 提交信息质量评分

### 9. 仓库与权限（3个）
- 仓库重要性评分
- 是否仓库 Owner
- 是否 Maintainer

### 10. 风险信号（6个）
- 涉及支付
- 涉及个人身份信息
- 涉及权限/鉴权
- 涉及安全模块
- 修改关键文件
- 有 TODO/FIXME

## 🔧 高级用法

### 训练单个模型

```bash
python models/lightgbm/train.py
python models/logistic/train.py
python models/svm/train.py
# ... 其他模型
```

### 生成测试集

```bash
python utils/create_test_set.py
```

会自动生成并标注测试数据，保存到 `data/test_sets/` 目录。

### 修改配置

编辑 `config.py` 文件可以修改：
- 特征列表
- 模型参数
- 自动合并阈值（默认 0.6）
- 强制拒绝条件

## 📈 评估指标

每个模型会输出以下指标：

- **准确率 (Accuracy)**: 整体预测正确的比例
- **精确率 (Precision)**: 预测为可合并中实际可合并的比例
- **召回率 (Recall)**: 实际可合并中被正确预测的比例
- **F1 分数**: 精确率和召回率的调和平均
- **AUC**: ROC 曲线下面积
- **混淆矩阵**: 详细的预测结果分布

## 📝 图表说明

### 单个模型训练结果 (training_results.png)
- 左上: 特征重要性 Top 15
- 右上: 预测概率分布
- 左下: 混淆矩阵
- 右下: 评估指标

### 单个模型测试结果 (xxx_test.png)
- 左上: 混淆矩阵
- 右上: 预测概率分布
- 左下: 评估指标
- 右下: 测试结果统计

### 模型对比 (comparison.png)
- 准确率对比
- AUC 对比
- ROC 曲线对比
- 精确率对比
- 召回率对比
- F1 对比
- 前 3 个模型的混淆矩阵

## ⚠️ 注意事项

1. **数据质量**：建议生成 500 条以上的数据，确保模型训练效果
2. **样本平衡**：自动标注会尽量保持正负样本平衡（40-60%）
3. **特征数量**：50 个特征是经过优化的，不建议随意增减
4. **阈值调整**：默认阈值 0.6，可根据实际需求在 `config.py` 中调整
5. **图表保存**：所有图表以文件形式保存，不会弹窗显示

## 📊 工作流程

```
1. 数据准备
   ├── 生成原始数据 (utils/generate_data.py)
   ├── 自动标注 (utils/auto_label.py)
   └── 人工标注 (1_data_labeling.py)

2. 模型训练
   ├── 批量训练 (train_all_models.py)
   │   ├── LightGBM
   │   ├── Logistic Regression
   │   ├── SVM
   │   ├── KNN
   │   ├── Random Forest
   │   └── MLP
   └── 查看各模型训练结果 (models/*/results/)

3. 模型测试与对比
   ├── 生成测试集 (utils/create_test_set.py)
   ├── 测试所有模型 (models/comparison/test_all_models.py)
   └── 查看对比结果 (models/comparison/results/)
```

## 🎯 目标

- ✅ 自动化 MR 评审流程
- ✅ 降低人工评审成本
- ✅ 提高合并效率
- ✅ 保证代码质量和安全性

## 📄 许可

MIT License
