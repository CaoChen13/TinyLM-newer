/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <math.h>
#include "model_params.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
I2C_HandleTypeDef hi2c1;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_I2C1_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
#define MPU_ADDR   (0x68 << 1)
#define WHO_AM_I   0x75
#define PWR_MGMT_1 0x6B
#define ACCEL_XOUT_H 0x3B
#define SMPLRT_DIV 0x19
#define CONFIG     0x1A

typedef struct {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
} mpu_raw_t;

// 数据缓冲区
static float win_buf[WIN_N][NUM_CHANNELS];
static int win_idx = 0;
static int win_count = 0;

int _write(int file, char *ptr, int len) {
  HAL_UART_Transmit(&huart2, (uint8_t*)ptr, len, 1000);
  return len;
}

static HAL_StatusTypeDef mpu_write_u8(uint8_t reg, uint8_t val) {
  return HAL_I2C_Mem_Write(&hi2c1, MPU_ADDR, reg, I2C_MEMADD_SIZE_8BIT, &val, 1, 100);
}

static HAL_StatusTypeDef mpu_read(uint8_t reg, uint8_t *buf, uint16_t len) {
  return HAL_I2C_Mem_Read(&hi2c1, MPU_ADDR, reg, I2C_MEMADD_SIZE_8BIT, buf, len, 100);
}

int mpu_read_raw(mpu_raw_t *d) {
  uint8_t buf[14];
  if (mpu_read(ACCEL_XOUT_H, buf, 14) != HAL_OK) return -1;
  
  d->ax = (int16_t)(buf[0] << 8 | buf[1]);
  d->ay = (int16_t)(buf[2] << 8 | buf[3]);
  d->az = (int16_t)(buf[4] << 8 | buf[5]);
  d->gx = (int16_t)(buf[8] << 8 | buf[9]);
  d->gy = (int16_t)(buf[10] << 8 | buf[11]);
  d->gz = (int16_t)(buf[12] << 8 | buf[13]);
  
  return 0;
}

void mpu_init(void) {
  HAL_Delay(100);
  mpu_write_u8(PWR_MGMT_1, 0x00);
  HAL_Delay(10);
  mpu_write_u8(CONFIG, 0x03);
  mpu_write_u8(SMPLRT_DIV, 19);  // 50Hz
}

// 把原始数据转换成物理单位并存入缓冲区
void mpu_add_sample(mpu_raw_t *raw) {
  win_buf[win_idx][0] = raw->ax / 16384.0f * 9.81f;
  win_buf[win_idx][1] = raw->ay / 16384.0f * 9.81f;
  win_buf[win_idx][2] = raw->az / 16384.0f * 9.81f;
  win_buf[win_idx][3] = raw->gx / 131.0f * 0.01745f;
  win_buf[win_idx][4] = raw->gy / 131.0f * 0.01745f;
  win_buf[win_idx][5] = raw->gz / 131.0f * 0.01745f;
  
  win_idx = (win_idx + 1) % WIN_N;
  if (win_count < WIN_N) win_count++;
}

// 提取特征
void extract_features(float *features) {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float sum = 0, sum_sq = 0;
    for (int i = 0; i < WIN_N; i++) {
      float v = (win_buf[i][ch] - NORM_MEAN[ch]) / NORM_STD[ch];
      sum += v;
      sum_sq += v * v;
    }
    float mean = sum / WIN_N;
    float var = sum_sq / WIN_N - mean * mean;
    features[ch] = mean;
    features[ch + 6] = sqrtf(var > 0 ? var : 0);
    features[ch + 12] = sum_sq / WIN_N;
  }
}

// 逻辑回归预测
int predict(float *features) {
  float best_score = -1e9f;
  int best = 0;
  
  for (int k = 0; k < NUM_CLASSES; k++) {
    float logit = LR_INTERCEPT[k];
    for (int i = 0; i < NUM_FEATURES; i++) {
      logit += features[i] * LR_COEF[k * NUM_FEATURES + i];
    }
    if (logit > best_score) {
      best_score = logit;
      best = k;
    }
  }
  return best;
}

// 模式: 0=识别, 1-6=采集对应类别
static int mode = 0;
static int last_collect_mode = 1;  // 上次采集的类别

// 检查串口命令
int check_command(void) {
  uint8_t ch;
  if (HAL_UART_Receive(&huart2, &ch, 1, 0) == HAL_OK) {
    if (ch == 'r' || ch == 'R') return -1;  // 切换模式
    if (ch >= '0' && ch <= '5') return ch - '0' + 1;  // 采集模式 1-6
    if (ch == 's' || ch == 'S') return -2;  // 停止采集
  }
  return 0;
}

// 检查按钮
int check_button(void) {
  static uint32_t last_press = 0;
  if (HAL_GPIO_ReadPin(B1_GPIO_Port, B1_Pin) == GPIO_PIN_RESET) {
    if (HAL_GetTick() - last_press > 300) {  // 防抖
      last_press = HAL_GetTick();
      return 1;
    }
  }
  return 0;
}

void print_help(void) {
  printf("\r\n=== Gesture System ===\r\n");
  printf("Commands:\r\n");
  printf("  r - Toggle mode (Recognition <-> Collect)\r\n");
  printf("  0-5 - Collect: still/tilt/shake/circle/tap/flip\r\n");
  printf("  s - Stop collecting\r\n");
  printf("  Button B1 - Toggle mode\r\n");
  printf("Current mode: %s\r\n\r\n", mode == 0 ? "Recognition" : "Collecting");
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_I2C1_Init();
  /* USER CODE BEGIN 2 */
  mpu_init();
  print_help();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  mpu_raw_t raw;
  float features[NUM_FEATURES];
  int sample_count = 0;
  
  // 稳定性输出
  int last_pred = -1;
  int same_count = 0;
  int stable_pred = -1;
  
  while (1)
  {
    // 检查命令
    int cmd = check_command();
    int btn = check_button();
    
    // 按钮或 r 键切换模式
    if (btn || cmd == -1) {
      if (mode == 0) {
        mode = last_collect_mode;
        printf("# Collecting: %s\r\n", LABEL_NAMES[mode-1]);
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
      } else {
        mode = 0;
        printf("# Mode: Recognition\r\n");
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
      }
    } else if (cmd == -2) {
      mode = 0;
      printf("# Stopped\r\n");
      HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
    } else if (cmd > 0) {
      mode = cmd;
      last_collect_mode = cmd;
      printf("# Collecting: %s\r\n", LABEL_NAMES[cmd-1]);
      HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
    }
    
    if (mpu_read_raw(&raw) == 0) {
      if (mode == 0) {
        // 识别模式
        mpu_add_sample(&raw);
        sample_count++;
        
        if (win_count >= WIN_N && sample_count >= HOP_N) {
          sample_count = 0;
          extract_features(features);
          int pred = predict(features);
          
          if (pred == last_pred) {
            same_count++;
          } else {
            same_count = 1;
            last_pred = pred;
          }
          
          if (same_count >= 3 && pred != stable_pred) {
            stable_pred = pred;
            printf("Gesture: %s\r\n", LABEL_NAMES[pred]);
          }
        }
      } else {
        // 采集模式
        printf("%s,%d,%d,%d,%d,%d,%d\r\n", 
               LABEL_NAMES[mode-1],
               raw.ax, raw.ay, raw.az,
               raw.gx, raw.gy, raw.gz);
      }
    }
    HAL_Delay(19);  // 约 50Hz
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
