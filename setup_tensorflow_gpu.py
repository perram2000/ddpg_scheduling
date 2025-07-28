# 创建检测脚本 check_gpu_detailed.py
import tensorflow as tf
import os

print("🔍 详细GPU环境检测")
print("=" * 50)

# TensorFlow版本
print(f"TensorFlow版本: {tf.__version__}")

# 检查CUDA支持
print(f"TensorFlow内置CUDA支持: {tf.test.is_built_with_cuda()}")
print(f"TensorFlow内置GPU支持: {tf.test.is_built_with_gpu_support()}")

# 检查物理设备
physical_devices = tf.config.list_physical_devices()
print(f"所有物理设备: {physical_devices}")

gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU设备数量: {len(gpu_devices)}")

# 检查环境变量
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

# 尝试GPU计算
try:
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = a + b
    print("✅ GPU计算测试: 成功")
except Exception as e:
    print(f"❌ GPU计算测试: 失败 - {e}")

# 检查CUDA库
try:
    import subprocess
    nvcc_result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if nvcc_result.returncode == 0:
        print("✅ NVCC可用")
        print(nvcc_result.stdout.split('\n')[3])  # CUDA版本行
    else:
        print("❌ NVCC不可用")
except:
    print("❌ NVCC不可用")