# EdgePick-Bot: 边缘端AI自动拾取机器人

基于 OpenMV + STM32 的单类目标识别与自动拾取闭环系统。

## 🎬 演示视频
👉 [演示视频](https://github.com/Georgeliin/EdgePick-Bot/blob/main/demo/demo.mp4)

## ✨ 性能指标
- 自建数据集准确率：92%
- 端侧推理帧率：15 FPS
- 系统响应时间：< 0.5s
- 自动拾取成功率：> 85%

## 🛠️ 技术栈
- **视觉端**：OpenMV, EdgeImpulse, MobileNetV2 (α=0.35)
- **控制端**：STM32 (串口通信, 运动控制)
- **开发语言**：Python, C/C++

## 📂 文件说明
- `main.py`：OpenMV 端主程序（视觉推理、目标跟踪、串口下发坐标）
- `model/`：量化模型文件
- `demo/`：项目演示视频

## 🚀 如何运行
1. 使用 OpenMV IDE 打开 `main.py`。
2. 将 `model` 文件夹中的模型文件存入 OpenMV 存储卡。
3. 连接 OpenMV 与 STM32 串口，上电运行。
