# TinyLM-newer
TinyML gesture recognition on STM32F411 (MPU6050) with training + deployment pipeline
# TinyML Gesture on STM32F411 (MPU6050)

Goal: run IMU gesture recognition on STM32F411 using a reproducible pipeline:
data -> windowing -> training -> int8 quantization -> deployment.

## Repo Structure
- firmware/ : STM32CubeIDE project (USART2 + I2C1, MPU6050)
- ml/       : data processing + training + tflite export/quant scripts
- docs/     : notes, demo screenshots/video links

## Milestones
- [ ] UART log works (Hello over USART2)
- [ ] I2C scan finds 0x68
- [ ] WHO_AM_I reads 0x68
- [ ] Train baseline classifier + confusion matrix
- [ ] Export int8 tflite and record size/accuracy drop
- [ ] On-device inference demo (video)

## How to Reproduce (WIP)
1. collect / prepare IMU data
2. run windowing script
3. train + evaluate
4. export tflite (float + int8)
