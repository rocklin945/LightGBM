"""
MR 风险评估模型配置文件（精简版）
目标：判断 MR 是否可以跳过人工评审，直接自动合并
"""

# ==================== 特征定义（精简到50个核心特征）====================

# 1. 代码变更规模特征（6个）
CODE_CHANGE_SIZE = [
    'total_changed_lines',        # 总改动行数
    'added_lines',                # 新增行数
    'deleted_lines',              # 删除行数
    'changed_files_count',        # 改动文件数
    'max_file_changed_lines',     # 单个文件最大改动行数
]

# 2. 文件类型特征（5个）
FILE_TYPE_FEATURES = [
    'code_files_count',           # 代码文件数
    'test_files_count',           # 测试文件数
    'has_only_test_change',       # 是否仅改动测试（0/1）
    'has_only_doc_change',        # 是否仅改动文档（0/1）
    'code_lines_ratio',           # 代码改动占比（0-1）
]

# 3. 代码复杂度特征（4个）
CODE_COMPLEXITY = [
    'modified_methods_count',     # 修改方法数
    'max_method_complexity',      # 最大方法圈复杂度
    'max_method_lines',           # 最大方法行数
    'max_nesting_level',          # 最大嵌套层级
]

# 4. 关键变更特征（5个）
CRITICAL_CHANGES = [
    'has_new_dependency',         # 是否新增依赖（0/1）
    'has_database_migration',     # 是否有数据库迁移（0/1）
    'has_api_change',             # 是否有 API 变更（0/1）
    'has_breaking_change',        # 有破坏性变更（0/1）
]

# 5. 测试质量特征（5个）
TEST_QUALITY = [
    'has_unit_test',              # 是否有单元测试（0/1）
    'unit_test_coverage',         # 单元测试覆盖率（0-100）
    'test_coverage_change',       # 覆盖率变化（百分点）
    'new_test_count',             # 新增测试数量
    'test_to_code_ratio',         # 测试代码与业务代码比例
]

# 6. CI/CD 质量特征（6个）
CI_QUALITY = [
    'ci_build_passed',            # 构建是否通过（0/1）
    'ci_unit_test_passed',        # 单元测试是否通过（0/1）
    'ci_lint_passed',             # 代码规范检查是否通过（0/1）
    'lint_issues_count',          # 代码规范问题数
    'security_vulnerabilities',   # 安全漏洞数
    'code_smells_count',          # 代码坏味道数
]

# 7. 作者质量特征（9个）
AUTHOR_QUALITY = [
    'author_total_mrs',           # 作者总MR数
    'author_mrs_last_30d',        # 作者最近30天MR数
    'author_merge_rate_30d',      # 作者最近30天合并率（0-1）
    'author_revert_rate_30d',     # 作者最近30天回滚率（0-1）
    'is_first_mr',                # 是否首个MR（0/1）
    'is_core_contributor',        # 是否核心贡献者（0/1）
    'author_days_in_repo',        # 作者在仓库天数
    'author_avg_mr_size',         # 作者平均MR大小
    'author_bug_fix_count_30d',   # 作者30天Bug修复数
]

# 8. 提交质量特征（4个）
COMMIT_QUALITY = [
    'commit_count',               # 提交次数
    'avg_commit_size',            # 平均提交大小
    'commit_message_quality',     # 提交信息质量评分（0-10）
]

# 9. 仓库与权限特征（4个）
REPO_PERMISSION = [
    'repo_importance_score',      # 仓库重要性评分（1-10）
    'is_repo_owner',              # 是否仓库Owner（0/1）
    'is_maintainer',              # 是否Maintainer（0/1）
]

# 10. 风险信号特征（7个）
RISK_SIGNALS = [
    'involves_payment',           # 涉及支付（0/1）
    'involves_pii',               # 涉及个人身份信息（0/1）
    'involves_auth',              # 涉及权限/鉴权（0/1）
    'involves_security',          # 涉及安全模块（0/1）
    'modifies_critical_file',     # 修改关键文件（0/1）
    'has_todo_or_fixme',          # 有TODO/FIXME（0/1）
]

# 所有特征列表
ALL_FEATURES = (
    CODE_CHANGE_SIZE +
    FILE_TYPE_FEATURES +
    CODE_COMPLEXITY +
    CRITICAL_CHANGES +
    TEST_QUALITY +
    CI_QUALITY +
    AUTHOR_QUALITY +
    COMMIT_QUALITY +
    REPO_PERMISSION +
    RISK_SIGNALS
)

# 目标变量
TARGET = 'can_auto_merge'

print(f"✓ 总特征数: {len(ALL_FEATURES)}")

# ==================== 特征说明 ====================

FEATURE_DESCRIPTIONS = {
    # 代码变更规模
    'total_changed_lines': '总改动行数',
    'added_lines': '新增行数',
    'deleted_lines': '删除行数',
    'changed_files_count': '改动文件数',
    'max_file_changed_lines': '单个文件最大改动行数',
    
    # 文件类型
    'code_files_count': '代码文件数',
    'test_files_count': '测试文件数',
    'has_only_test_change': '仅改动测试',
    'has_only_doc_change': '仅改动文档',
    'code_lines_ratio': '代码改动占比',
    
    # 代码复杂度
    'modified_methods_count': '修改方法数',
    'max_method_complexity': '最大方法圈复杂度',
    'max_method_lines': '最大方法行数',
    'max_nesting_level': '最大嵌套层级',
    
    # 关键变更
    'has_new_dependency': '新增依赖',
    'has_database_migration': '数据库迁移',
    'has_api_change': 'API变更',
    'has_breaking_change': '破坏性变更',
    
    # 测试质量
    'has_unit_test': '有单元测试',
    'unit_test_coverage': '单元测试覆盖率',
    'test_coverage_change': '覆盖率变化',
    'new_test_count': '新增测试数量',
    'test_to_code_ratio': '测试代码比例',
    
    # CI/CD质量
    'ci_build_passed': 'CI构建通过',
    'ci_unit_test_passed': 'CI单测通过',
    'ci_lint_passed': 'CI规范检查通过',
    'lint_issues_count': '代码规范问题数',
    'security_vulnerabilities': '安全漏洞数',
    'code_smells_count': '代码坏味道数',
    
    # 作者质量
    'author_total_mrs': '作者总MR数',
    'author_mrs_last_30d': '作者30天MR数',
    'author_merge_rate_30d': '作者30天合并率',
    'author_revert_rate_30d': '作者30天回滚率',
    'is_first_mr': '首个MR',
    'is_core_contributor': '核心贡献者',
    'author_days_in_repo': '作者在仓库天数',
    'author_avg_mr_size': '作者平均MR大小',
    'author_bug_fix_count_30d': '作者30天Bug修复数',
    
    # 提交质量
    'commit_count': '提交次数',
    'avg_commit_size': '平均提交大小',
    'commit_message_quality': '提交信息质量',
    
    # 仓库权限
    'repo_importance_score': '仓库重要性',
    'is_repo_owner': '仓库Owner',
    'is_maintainer': 'Maintainer',
    
    # 风险信号
    'involves_payment': '涉及支付',
    'involves_pii': '涉及PII',
    'involves_auth': '涉及权限',
    'involves_security': '涉及安全',
    'modifies_critical_file': '修改关键文件',
    'has_todo_or_fixme': 'TODO/FIXME',
    
    # 目标变量
    'can_auto_merge': '可自动合并',
}

# ==================== 模型参数 ====================

LIGHTGBM_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,       # 特征少了，可以用更多
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 10,       # 数据多了，可以降低
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'seed': 2024,
    'verbose': -1,
}

# ==================== 风险阈值 ====================

# 强制拒绝自动合并的条件（更宽松）
FORCE_REJECT_CONDITIONS = {
    'involves_payment': 1,           # 涉及支付
    'ci_build_passed': 0,            # 构建未通过
}

# 自动合并阈值（降低到0.6，更容易通过）
AUTO_MERGE_THRESHOLD = 0.6

# ==================== 文件路径 ====================

DATA_DIR = 'd:/LightGBM/data'
MODEL_DIR = 'd:/LightGBM/models'

RAW_DATA_PATH = f'{DATA_DIR}/raw_mr_data.csv'
LABELED_DATA_PATH = f'{DATA_DIR}/labeled_data.csv'
TRAIN_DATA_PATH = f'{DATA_DIR}/train_data.csv'
TEST_DATA_PATH = f'{DATA_DIR}/test_data.csv'

MODEL_PATH = f'{MODEL_DIR}/mr_risk_model.txt'
FEATURE_IMPORTANCE_PATH = f'{MODEL_DIR}/feature_importance.csv'
