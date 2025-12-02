# quick_check.py
import subprocess
import sys

def quick_dependency_check():
    """快速依赖检查"""
    print("🚀 快速依赖冲突检查")
    print("=" * 50)
    
    # 1. 运行 pip check
    print("1. 运行 pip check...")
    result = subprocess.run([sys.executable, "-m", "pip", "check"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ pip check: 无冲突")
    else:
        print("❌ pip check 发现冲突:")
        print(result.stdout)
    
    # 2. 检查关键包版本
    print("\n2. 检查关键包版本...")
    packages = ["tensorflow", "torch", "transformers", "flatbuffers", "tf2onnx", "numpy"]
    
    for pkg in packages:
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "show", pkg], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # 提取版本信息
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':', 1)[1].strip()
                        print(f"   {pkg}: {version}")
                        break
            else:
                print(f"   {pkg}: 未安装")
        except:
            print(f"   {pkg}: 检查失败")
    
    # 3. 测试导入
    print("\n3. 测试包导入...")
    test_imports = [
        "import tensorflow as tf",
        "import torch", 
        "import transformers",
        "import flatbuffers",
        "import numpy as np"
    ]
    
    for import_stmt in test_imports:
        try:
            exec(import_stmt)
            pkg = import_stmt.split()[-1]
            print(f"   ✅ {pkg}: 导入成功")
        except Exception as e:
            pkg = import_stmt.split()[-1]
            print(f"   ❌ {pkg}: 导入失败 - {e}")

if __name__ == "__main__":
    quick_dependency_check()