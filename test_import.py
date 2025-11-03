import sys
sys.path.insert(0, 'src')

try:
    from che.core.ecosystem import Ecosystem
    print('Ecosystem类可以正常导入')
    
    from REAL_SCIENTIFIC_EXPERIMENT_GEMMA3_IMPROVED import validate_cognitive_independence_correlation
    print('验证函数可以正常导入')
    
    print("所有导入测试通过！")
except Exception as e:
    print(f"导入失败: {e}")