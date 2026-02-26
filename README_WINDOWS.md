# Windows 启动指南 (Windows Startup Guide)

本项目采用 **FastAPI 后端 + PyQt5 前端** 的架构。启动系统需要开启两个进程：API 服务器和 GUI 界面。

## 1. 环境准备 (Environment Setup)

在 Windows 搜索框输入 `cmd` 或 `PowerShell` 并打开。

```bash
# 1. 进入项目根目录 (假设在 D:\attentionSleep)
cd /d D:\attentionSleep

# 2. 创建虚拟环境 (仅首次需要)
python -m venv venv

# 3. 激活虚拟环境
venv\Scripts\activate

# 4. 安装依赖
pip install -r requirements.txt
```

## 2. 启动步骤 (Startup Steps)

### 第一步：启动 API 后端服务器
API 服务器负责运行 AI 模型进行疲劳检测。

1. 打开一个终端控制台。
2. 激活环境并运行：
```bash
venv\Scripts\activate
python api/server.py
```
*看到 `Uvicorn running on http://0.0.0.0:8000` 表示启动成功。*

### 第二步：启动 GUI 前端程序
GUI 负责显示摄像头画面和检测结果。

1. **另外打开一个** 终端控制台。
2. 激活环境并运行：
```bash
venv\Scripts\activate
python main.py
```

## 3. 注意事项 (Notes)

- **模型文件**：确保 `models/checkpoints/best_model.pth` 文件存在。
- **摄像头**：程序启动后会自动调用默认摄像头（ID 0）。
- **人脸识别组件**：首次运行 API 时会自动加载 MediaPipe 模型，请确保网络连接。
- **防火墙**：如果提示网络权限，请允许应用访问，因为前端和后端通过 HTTP 协议通信。

---
*Created by Antigravity AI Assistant*
