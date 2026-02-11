"""
批量删除所有模型训练结果
"""
import os
import shutil

def delete_file(filepath):
    """删除文件"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"  ✓ 删除文件: {filepath}")
            return True
    except Exception as e:
        print(f"  ✗ 删除失败: {filepath} - {e}")
        return False
    return False

def delete_directory(dirpath):
    """删除目录"""
    try:
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
            print(f"  ✓ 删除目录: {dirpath}")
            return True
    except Exception as e:
        print(f"  ✗ 删除失败: {dirpath} - {e}")
        return False
    return False

def clean_model_results():
    """清理所有模型结果"""
    print("=" * 80)
    print("批量删除所有模型训练结果")
    print("=" * 80)
    
    deleted_count = 0
    
    # 模型列表
    models = ['lightgbm', 'logistic', 'svm', 'knn', 'random_forest', 'mlp']
    
    # 1. 删除各个模型的训练文件（不删除results文件夹）
    print("\n[1] 清理各模型训练文件...")
    for model_name in models:
        model_dir = f'models/{model_name}'
        
        if not os.path.exists(model_dir):
            continue
        
        print(f"\n  清理 {model_name.upper()} 模型:")
        
        # 删除模型文件
        if model_name == 'lightgbm':
            if delete_file(f'{model_dir}/model.txt'):
                deleted_count += 1
        else:
            if delete_file(f'{model_dir}/model.pkl'):
                deleted_count += 1
            if delete_file(f'{model_dir}/scaler.pkl'):
                deleted_count += 1
        
        # 删除 results 目录
        results_dir = f'{model_dir}/results'
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if delete_file(os.path.join(results_dir, file)):
                    deleted_count += 1
    
    
    # 3. 询问是否删除数据文件
    print("\n[3] 数据文件...")
    delete_data = input("  是否删除数据文件？(y/n): ").strip().lower()
    
    if delete_data == 'y':
        data_files = [
            'data/labeled_data.csv',
            'data/train_data.csv',
            'data/test_data.csv',
        ]
        
        for file in data_files:
            if delete_file(file):
                deleted_count += 1
        
        # 删除测试集目录
        test_sets_dir = 'data/test_sets'
        if os.path.exists(test_sets_dir):
            for file in os.listdir(test_sets_dir):
                if delete_file(os.path.join(test_sets_dir, file)):
                    deleted_count += 1
        
        # 询问是否删除原始数据
        delete_raw = input("  是否删除原始数据 (raw_mr_data.csv)？(y/n): ").strip().lower()
        if delete_raw == 'y':
            if delete_file('data/raw_mr_data.csv'):
                deleted_count += 1
    
    print("\n" + "=" * 80)
    print(f"清理完成！共删除 {deleted_count} 个文件")
    print("=" * 80)
    
    print("\n下一步:")
    print("  1. 生成新数据: python utils/generate_data.py")
    print("  2. 自动标注: python utils/auto_label.py")
    print("  3. 批量训练: python train_all_models.py")

def main():
    """主函数"""
    print("\n⚠️  警告: 此操作将删除所有模型训练结果！")
    confirm = input("确认继续？(y/n): ").strip().lower()
    
    if confirm == 'y':
        clean_model_results()
    else:
        print("已取消")

if __name__ == '__main__':
    main()
