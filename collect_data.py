"""
MPU6050 数据采集脚本
用法: python collect_data.py COM端口号
例如: python collect_data.py COM3
"""

import serial
import sys
import os
from datetime import datetime

# 类别
LABELS = ['still', 'tilt', 'shake', 'circle', 'tap', 'flip']

def main():
    if len(sys.argv) < 2:
        print("用法: python collect_data.py COM端口号")
        print("例如: python collect_data.py COM3")
        sys.exit(1)
    
    port = sys.argv[1]
    
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    # 打开串口
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
        print(f"已连接到 {port}")
    except Exception as e:
        print(f"无法打开串口: {e}")
        sys.exit(1)
    
    # 数据文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/mpu6050_{timestamp}.csv"
    
    print(f"\n数据将保存到: {filename}")
    print("\n操作说明:")
    print("  输入 0-5 开始录制对应动作")
    print("  输入 s 停止录制")
    print("  输入 q 退出程序")
    print("\n动作列表:")
    for i, label in enumerate(LABELS):
        print(f"  {i} - {label}")
    print()
    
    # 写入 CSV 头
    with open(filename, 'w') as f:
        f.write("label,ax,ay,az,gx,gy,gz\n")
    
    sample_count = 0
    
    try:
        while True:
            # 检查用户输入
            try:
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8', errors='ignore')
                    if key == 'q':
                        break
                    elif key in '012345s':
                        ser.write(key.encode())
                        print(f"发送命令: {key}")
            except ImportError:
                # Linux/Mac
                import select
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        break
                    elif key in '012345s':
                        ser.write(key.encode())
                        print(f"发送命令: {key}")
            
            # 读取串口数据
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    # 跳过注释行
                    if line.startswith('#'):
                        print(line)
                    # CSV 数据行
                    elif ',' in line and not line.startswith('='):
                        parts = line.split(',')
                        if len(parts) == 7 and parts[0] in LABELS:
                            with open(filename, 'a') as f:
                                f.write(line + '\n')
                            sample_count += 1
                            if sample_count % 50 == 0:
                                print(f"已采集 {sample_count} 个样本")
                    else:
                        print(line)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        ser.close()
        print(f"\n采集完成! 共 {sample_count} 个样本")
        print(f"数据已保存到: {filename}")

if __name__ == '__main__':
    main()
