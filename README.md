# STM32 MPU6050 手势识别系统

基于 STM32F411RE + MPU6050 的 TinyML 手势识别项目。

## 硬件
- NUCLEO-F411RE 开发板
- MPU6050 六轴传感器 (I2C: PB8=SCL, PB9=SDA)

## 功能
识别 6 种手势：
- still (静止)
- tilt (倾斜)
- shake (摇晃)
- circle (画圈)
- tap (敲击)
- flip (翻转)

## 技术参数
- 采样率: 50Hz
- 窗口: 25 samples (0.5s)
- 步长: 5 samples (0.1s)
- 特征: 18维 (6通道 × mean/std/energy)
- 模型: Logistic Regression

## 使用方法

### 1. 数据采集
```bash
python collect_data.py COM4
```
按 0-5 录制对应手势，按 s 停止，按 q 退出。

### 2. 模型训练
```bash
python train_model.py
```
自动导出参数到 `Inc/model_params.h`。

### 3. 固件烧录
用 STM32CubeIDE 编译烧录。

### 4. 运行
- 按 B1 按钮或发送 'r' 切换 采集/识别 模式
- LED2 亮 = 采集模式，灭 = 识别模式

## 项目结构
```
├── Src/main.c           # 主程序
├── Inc/model_params.h   # 模型参数
├── train_model.py       # 训练脚本
├── collect_data.py      # 数据采集脚本
├── data/                # 采集的数据
└── docs/                # 说明文档
```

## License
MIT
