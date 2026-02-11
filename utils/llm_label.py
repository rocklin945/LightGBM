"""
使用大模型API进行自动标注
控制token消耗，批量处理数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
import json
from config import *
from collections import Counter

# API配置 - 从环境变量或配置文件读取
API_KEY = os.environ.get('LLM_API_KEY', '')
BASE_URL = os.environ.get('LLM_BASE_URL', 'https://coding.qunhequnhe.com')

# 如果环境变量没有设置，尝试从配置文件读取
if not API_KEY:
    config_file = os.path.join(os.path.dirname(__file__), '.llm_config')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                if line.startswith('API_KEY='):
                    API_KEY = line.split('=', 1)[1].strip()
                elif line.startswith('BASE_URL='):
                    BASE_URL = line.split('=', 1)[1].strip()

API_ENDPOINT = f"{BASE_URL}/v1/chat/completions"

# Token统计
token_stats = {
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_tokens': 0,
    'api_calls': 0
}


def create_prompt(mr_data):
    """
    创建精简的prompt，减少token消耗
    
    参数:
        mr_data: MR数据字典
    
    返回:
        prompt字符串
    """
    # 只包含关键特征，精简描述
    prompt = f"""判断此MR是否可自动合并(0=否,1=是):

代码规模: {mr_data['total_changed_lines']}行, {mr_data['changed_files_count']}文件
作者: 总MR={mr_data['author_total_mrs']}, 合并率={mr_data['author_merge_rate_30d']:.2f}, 回滚率={mr_data['author_revert_rate_30d']:.3f}
CI: 构建={'通过' if mr_data['ci_build_passed'] else '失败'}, 测试={'通过' if mr_data['ci_unit_test_passed'] else '失败'}, 规范={'通过' if mr_data['ci_lint_passed'] else '失败'}
测试: 覆盖率={mr_data['unit_test_coverage']:.1f}%, 新增={mr_data['new_test_count']}个
质量: 规范问题={mr_data['lint_issues_count']}, 安全漏洞={mr_data['security_vulnerabilities']}, 坏味道={mr_data['code_smells_count']}
风险: 支付={mr_data['involves_payment']}, PII={mr_data['involves_pii']}, 权限={mr_data['involves_auth']}, 破坏性={mr_data['has_breaking_change']}
权限: Owner={mr_data['is_repo_owner']}, Maintainer={mr_data['is_maintainer']}, 核心={mr_data['is_core_contributor']}

仅回复: {{"label": 0或1, "reason": "原因(10字内)"}}"""
    
    return prompt


def call_llm_api(prompt, model="gpt-4o-mini"):
    """
    调用大模型API
    
    参数:
        prompt: 提示词
        model: 模型名称
    
    返回:
        response字典或None
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是MR风险评估专家。分析MR特征，判断是否可自动合并。严格按JSON格式回复。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,  # 降低随机性
        "max_tokens": 100,   # 限制输出长度
        "response_format": {"type": "json_object"}  # 强制JSON输出
    }
    
    try:
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # 更新token统计
            if 'usage' in result:
                token_stats['prompt_tokens'] += result['usage'].get('prompt_tokens', 0)
                token_stats['completion_tokens'] += result['usage'].get('completion_tokens', 0)
                token_stats['total_tokens'] += result['usage'].get('total_tokens', 0)
            token_stats['api_calls'] += 1
            
            return result
        else:
            print(f"  ✗ API调用失败: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"  ✗ API调用异常: {e}")
        return None


def parse_llm_response(response):
    """
    解析大模型返回结果
    
    参数:
        response: API返回的字典
    
    返回:
        (label, reason) 或 (None, None)
    """
    try:
        if not response or 'choices' not in response:
            return None, None
        
        content = response['choices'][0]['message']['content']
        result = json.loads(content)
        
        label = int(result.get('label', -1))
        reason = result.get('reason', '未知原因')
        
        if label in [0, 1]:
            return label, f"LLM:{reason}"
        else:
            return None, None
            
    except Exception as e:
        print(f"  ✗ 解析响应失败: {e}")
        return None, None


def llm_label_batch(df, batch_size=5, skip_labeled=True):
    """
    使用大模型批量标注数据
    
    参数:
        df: 输入DataFrame
        batch_size: 每批处理数量（控制API调用频率）
        skip_labeled: 是否跳过已标注数据
    
    返回:
        df: 标注后的DataFrame
        stats: 标注统计
    """
    llm_labeled = 0
    failed = 0
    reasons = []
    
    # 获取未标注数据
    if skip_labeled:
        unlabeled_indices = df[df[TARGET] == -1].index.tolist()
    else:
        unlabeled_indices = df.index.tolist()
    
    total_unlabeled = len(unlabeled_indices)
    
    if total_unlabeled == 0:
        print("没有需要标注的数据")
        return df, {
            'llm_labeled': 0,
            'failed': 0,
            'can_merge': len(df[df[TARGET] == 1]),
            'cannot_merge': len(df[df[TARGET] == 0]),
            'unlabeled': 0,
            'reasons': []
        }
    
    print(f"\n准备标注 {total_unlabeled} 条数据...")
    print(f"批量大小: {batch_size} (控制API调用频率)")
    
    # 批量处理
    for i in range(0, total_unlabeled, batch_size):
        batch_indices = unlabeled_indices[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_unlabeled + batch_size - 1) // batch_size
        
        print(f"\n处理批次 {batch_num}/{total_batches} ({len(batch_indices)} 条)...")
        
        for idx in batch_indices:
            row = df.loc[idx]
            
            # 创建prompt
            prompt = create_prompt(row.to_dict())
            
            # 调用API
            response = call_llm_api(prompt)
            
            if response:
                label, reason = parse_llm_response(response)
                
                if label is not None:
                    df.at[idx, TARGET] = label
                    llm_labeled += 1
                    reasons.append((label, reason))
                    print(f"  ✓ MR-{idx+1:05d}: {'可合并' if label == 1 else '不可合并'} - {reason}")
                else:
                    failed += 1
                    print(f"  ✗ MR-{idx+1:05d}: 解析失败")
            else:
                failed += 1
                print(f"  ✗ MR-{idx+1:05d}: API调用失败")
    
    # 统计
    n_can = len(df[df[TARGET] == 1])
    n_cannot = len(df[df[TARGET] == 0])
    n_unlabeled = len(df[df[TARGET] == -1])
    
    stats = {
        'llm_labeled': llm_labeled,
        'failed': failed,
        'can_merge': n_can,
        'cannot_merge': n_cannot,
        'unlabeled': n_unlabeled,
        'reasons': reasons
    }
    
    return df, stats


def print_statistics(stats, total):
    """打印标注统计"""
    print("\n" + "=" * 80)
    print("LLM 标注统计")
    print("=" * 80)
    
    print(f"\n本次LLM标注:")
    print(f"  成功: {stats['llm_labeled']:4d} 条")
    print(f"  失败: {stats['failed']:4d} 条")
    
    print(f"\n当前状态:")
    print(f"  可以自动合并:   {stats['can_merge']:4d} ({stats['can_merge']/total*100:5.1f}%)")
    print(f"  不可以自动合并: {stats['cannot_merge']:4d} ({stats['cannot_merge']/total*100:5.1f}%)")
    print(f"  仍未标注:       {stats['unlabeled']:4d} ({stats['unlabeled']/total*100:5.1f}%)")
    
    if stats['can_merge'] + stats['cannot_merge'] > 0:
        merge_ratio = stats['can_merge'] / (stats['can_merge'] + stats['cannot_merge']) * 100
        print(f"\n  自动合并比例: {merge_ratio:.1f}%")
    
    # Token消耗统计
    print("\n" + "-" * 80)
    print("Token 消耗统计")
    print("-" * 80)
    print(f"  API调用次数:    {token_stats['api_calls']}")
    print(f"  Prompt Tokens:  {token_stats['prompt_tokens']}")
    print(f"  Completion Tokens: {token_stats['completion_tokens']}")
    print(f"  总Token消耗:    {token_stats['total_tokens']}")
    
    if token_stats['api_calls'] > 0:
        avg_tokens = token_stats['total_tokens'] / token_stats['api_calls']
        print(f"  平均每次调用:   {avg_tokens:.1f} tokens")
    
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
    print("大模型API自动标注工具")
    print("=" * 80)
    
    # 检查API配置
    if not API_KEY:
        print("\n错误: 未配置 API Key")
        print("\n请通过以下方式之一配置:")
        print("  1. 设置环境变量:")
        print("     export LLM_API_KEY='your-api-key'")
        print("     export LLM_BASE_URL='https://coding.qunhequnhe.com'")
        print("\n  2. 创建配置文件 utils/.llm_config:")
        print("     API_KEY=your-api-key")
        print("     BASE_URL=https://coding.qunhequnhe.com")
        print("\n  注意: 配置文件已添加到 .gitignore，不会被提交")
        return
    
    print(f"\nAPI配置:")
    print(f"  Base URL: {BASE_URL}")
    print(f"  API Key: {API_KEY[:10]}...")
    
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
        print("\n所有数据已标注完成！")
        return
    
    # 询问批量大小
    print(f"\n⚠️  注意: 每次API调用约消耗 200-300 tokens")
    print(f"  预计总消耗: {unlabeled_count * 250} tokens")
    
    try:
        batch_input = input("\n批量大小 (默认5，越小越慢但更稳定): ").strip()
        batch_size = int(batch_input) if batch_input else 5
    except:
        batch_size = 5
    
    confirm = input(f"\n确认使用LLM标注 {unlabeled_count} 条数据？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    # LLM标注
    print("\n开始LLM标注...")
    df, stats = llm_label_batch(df, batch_size=batch_size, skip_labeled=True)
    
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
    if stats['unlabeled'] > 0:
        print(f"\n还有 {stats['unlabeled']} 条数据未标注")
        print(f"可以:")
        print(f"  1. 再次运行本脚本继续LLM标注")
        print(f"  2. 运行规则标注: python utils/auto_label.py")
    else:
        print("\n所有数据已标注完成！")
        print(f"下一步:")
        print(f"  批量训练: python train_all_models.py")


if __name__ == '__main__':
    main()
