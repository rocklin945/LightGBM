"""
使用大模型API进行自动标注
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
import json
import time
import urllib3
from config import *

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API配置
API_KEY = ''
BASE_URL = 'https://coding.qunhequnhe.com'
MODEL_NAME = 'qwen3-coder-30b'

# 读取配置文件
config_file = os.path.join(os.path.dirname(__file__), '.llm_config')
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if line.startswith('API_KEY='):
                API_KEY = line.split('=', 1)[1].strip()
            elif line.startswith('BASE_URL='):
                BASE_URL = line.split('=', 1)[1].strip()
            elif line.startswith('MODEL='):
                MODEL_NAME = line.split('=', 1)[1].strip()

# 构建API端点
BASE_URL = BASE_URL.rstrip('/')
if BASE_URL.endswith('/v1'):
    API_ENDPOINT = f"{BASE_URL}/chat/completions"
else:
    API_ENDPOINT = f"{BASE_URL}/v1/chat/completions"


def create_prompt(mr_data):
    """创建prompt"""
    prompt = f"""判断MR是否可自动合并(0=否,1=是):

[规模] 总行:{mr_data['total_changed_lines']}, 增:{mr_data['added_lines']}, 删:{mr_data['deleted_lines']}, 文件:{mr_data['changed_files_count']}, 单文件最大:{mr_data['max_file_changed_lines']}
[文件] 代码:{mr_data['code_files_count']}, 测试:{mr_data['test_files_count']}, 纯测试:{mr_data['has_only_test_change']}, 纯文档:{mr_data['has_only_doc_change']}, 代码占比:{mr_data['code_lines_ratio']:.2f}
[复杂度] 方法数:{mr_data['modified_methods_count']}, 圈复杂度:{mr_data['max_method_complexity']:.0f}, 方法行数:{mr_data['max_method_lines']}, 嵌套:{mr_data['max_nesting_level']}
[关键] 新依赖:{mr_data['has_new_dependency']}, 数据库迁移:{mr_data['has_database_migration']}, API变更:{mr_data['has_api_change']}, 破坏性:{mr_data['has_breaking_change']}
[测试] 有单测:{mr_data['has_unit_test']}, 覆盖率:{mr_data['unit_test_coverage']:.1f}%, 覆盖变化:{mr_data['test_coverage_change']:.1f}, 新测试:{mr_data['new_test_count']}, 测试比:{mr_data['test_to_code_ratio']:.2f}
[CI] 构建:{mr_data['ci_build_passed']}, 单测:{mr_data['ci_unit_test_passed']}, 规范:{mr_data['ci_lint_passed']}, 规范问题:{mr_data['lint_issues_count']}, 漏洞:{mr_data['security_vulnerabilities']}, 坏味道:{mr_data['code_smells_count']}
[作者] 总MR:{mr_data['author_total_mrs']}, 30dMR:{mr_data['author_mrs_last_30d']}, 合并率:{mr_data['author_merge_rate_30d']:.2f}, 回滚率:{mr_data['author_revert_rate_30d']:.3f}, 首个MR:{mr_data['is_first_mr']}, 核心:{mr_data['is_core_contributor']}, 天数:{mr_data['author_days_in_repo']}, 平均规模:{mr_data['author_avg_mr_size']}, Bug修复:{mr_data['author_bug_fix_count_30d']}
[提交] 次数:{mr_data['commit_count']}, 平均大小:{mr_data['avg_commit_size']}, 信息质量:{mr_data['commit_message_quality']:.1f}
[仓库] 重要性:{mr_data['repo_importance_score']:.1f}, Owner:{mr_data['is_repo_owner']}, Maintainer:{mr_data['is_maintainer']}
[风险] 支付:{mr_data['involves_payment']}, PII:{mr_data['involves_pii']}, 权限:{mr_data['involves_auth']}, 安全:{mr_data['involves_security']}, 关键文件:{mr_data['modifies_critical_file']}, TODO:{mr_data['has_todo_or_fixme']}

回复JSON: {{"label": 0或1, "reason": "原因(15字内)"}}"""
    return prompt


def call_llm_api(prompt, max_retries=3):
    """调用大模型API，失败自动重试"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是MR风险评估专家。分析MR特征，判断是否可自动合并。严格按JSON格式回复。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 100
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                if attempt < max_retries - 1:
                    print(f"  ⚠ API错误 {response.status_code}, 重试 {attempt + 1}/{max_retries - 1}")
                    time.sleep(2)
                else:
                    print(f"  ✗ API错误: {response.status_code}")
                    return None
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ 请求异常, 重试 {attempt + 1}/{max_retries - 1}")
                time.sleep(2)
            else:
                print(f"  ✗ 请求失败: {str(e)[:50]}")
                return None
    
    return None


def parse_llm_response(response):
    """解析API响应"""
    try:
        content = response['choices'][0]['message']['content'].strip()
        
        # 清理markdown格式
        if content.startswith('```'):
            content = content.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(content)
        label = int(result.get('label', -1))
        reason = result.get('reason', '未知')
        
        if label in [0, 1]:
            return label, reason
        return None, None
    except:
        return None, None


def llm_label_batch(df, interval=0.5):
    """批量标注所有数据"""
    total = len(df)
    print(f"\n开始标注 {total} 条数据 (请求间隔: {interval}秒)...")
    
    success = 0
    failed = 0
    
    for idx in df.index:
        row = df.loc[idx]
        prompt = create_prompt(row.to_dict())
        response = call_llm_api(prompt)
        
        if response:
            label, reason = parse_llm_response(response)
            if label is not None:
                df.at[idx, TARGET] = label
                success += 1
                print(f"  [{idx+1}/{total}] MR-{idx+1:05d}: {'✓ 可合并' if label == 1 else '✗ 不可合并'} - {reason}")
            else:
                failed += 1
                df.at[idx, TARGET] = -1
                print(f"  [{idx+1}/{total}] MR-{idx+1:05d}: 解析失败")
        else:
            failed += 1
            df.at[idx, TARGET] = -1
            print(f"  [{idx+1}/{total}] MR-{idx+1:05d}: API失败")
        
        # 每次请求后延迟
        if idx < total - 1:  # 最后一条不需要延迟
            time.sleep(interval)
    
    print(f"\n标注完成: 成功 {success} 条, 失败 {failed} 条")
    return df


def main():
    """主函数"""
    print("=" * 80)
    print("大模型API自动标注工具")
    print("=" * 80)
    
    # 检查配置
    if not API_KEY:
        print("\n错误: 未配置API Key")
        print("请编辑 utils/.llm_config 文件")
        return
    
    print(f"\nAPI配置: {BASE_URL}")
    print(f"模型: {MODEL_NAME}")
    
    # 加载数据
    if not os.path.exists(RAW_DATA_PATH):
        print(f"\n错误: 数据文件不存在 {RAW_DATA_PATH}")
        return
    
    df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')
    print(f"\n加载数据: {len(df)} 条")
    
    # 请求间隔
    try:
        interval_input = input("\n请求间隔/秒 (默认0.5): ").strip()
        interval = float(interval_input) if interval_input else 0.5
    except:
        interval = 0.5
    
    # 开始标注
    df = llm_label_batch(df, interval)
    
    # 统计结果
    can_merge = len(df[df[TARGET] == 1])
    cannot_merge = len(df[df[TARGET] == 0])
    failed = len(df[df[TARGET] == -1])
    
    print(f"\n标注结果:")
    print(f"  可合并: {can_merge}")
    print(f"  不可合并: {cannot_merge}")
    print(f"  失败: {failed}")
    
    # 保存标注数据（不修改原始数据）
    df.to_csv(LABELED_DATA_PATH, index=False, encoding='utf-8-sig')
    print(f"\n已保存到: {LABELED_DATA_PATH}")
    print(f"原始数据未修改: {RAW_DATA_PATH}")


if __name__ == '__main__':
    main()
