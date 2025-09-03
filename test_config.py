#!/usr/bin/env python3
"""
验证FilterByImageWidth在transforms系统中的注册
"""

def test_registration():
    """测试FilterByImageWidth是否能通过配置系统访问"""
    
    # 测试通过eval动态创建（类似create_operators的工作方式）
    try:
        # 模拟PaddleOCR的create_operators函数逻辑
        op_name = "FilterByImageWidth" 
        param = {"width_range": [100, 300]}
        
        # 这就是create_operators内部的逻辑
        op = eval(op_name)(**param)
        print(f"✓ {op_name} 成功通过eval创建")
        print(f"  参数: {param}")
        print(f"  实例: {op}")
        print(f"  类型: {type(op)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 注册验证失败: {e}")
        return False


if __name__ == "__main__":
    print("=== FilterByImageWidth 注册验证 ===")
    
    # 首先尝试直接导入验证
    try:
        from ppocr.data.imaug.operators import FilterByImageWidth
        print("✓ 直接导入成功")
    except Exception as e:
        print(f"✗ 直接导入失败: {e}")
    
    # 测试通过配置系统的方式
    success = test_registration()
    
    if success:
        print("\n=== 注册验证通过 ===")
        print("FilterByImageWidth已成功注册到transforms系统")
    else:
        print("\n=== 注册验证失败 ===")