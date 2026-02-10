"""
批量训练所有模型
"""
import subprocess
import os
import sys

MODEL_SCRIPTS = [
    'models/lightgbm/train.py',
    'models/logistic/train.py',
    'models/svm/train.py',
    'models/knn/train.py',
    'models/random_forest/train.py',
    'models/mlp/train.py',
]

def main():
    """批量训练所有模型"""
    print("=" * 80)
    print("批量训练所有模型")
    print("=" * 80)
    
    success_count = 0
    failed_models = []
    
    for i, script in enumerate(MODEL_SCRIPTS, 1):
        model_name = script.split('/')[1]
        print(f"\n{'='*80}")
        print(f"[{i}/{len(MODEL_SCRIPTS)}] 训练 {model_name.upper()} 模型")
        print(f"{'='*80}\n")
        
        try:
            result = subprocess.run(
                [sys.executable, script],
                check=True,
                capture_output=False
            )
            
            if result.returncode == 0:
                success_count += 1
                print(f"\n✓ {model_name} 训练成功")
            else:
                failed_models.append(model_name)
                print(f"\n✗ {model_name} 训练失败")
                
        except subprocess.CalledProcessError as e:
            failed_models.append(model_name)
            print(f"\n✗ {model_name} 训练失败: {e}")
        except Exception as e:
            failed_models.append(model_name)
            print(f"\n✗ {model_name} 训练失败: {e}")
    
    # 汇总
    print("\n" + "=" * 80)
    print("训练完成汇总")
    print("=" * 80)
    print(f"\n成功: {success_count}/{len(MODEL_SCRIPTS)}")
    
    if failed_models:
        print(f"失败: {', '.join(failed_models)}")
    else:
        print("所有模型训练成功！")
    
    print("\n下一步:")
    print("  运行: python models/comparison/test_all_models.py")


if __name__ == '__main__':
    main()
