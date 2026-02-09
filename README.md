# Mario Face Swap

基于经典 Super Mario Bros Level 1 的 AI 人脸替换游戏。拍一张照片，AI 将你的脸转化为像素风格并替换马里奥的头部，体验专属于你的马里奥冒险！

![screenshot](Mario-Level-1/screenshot.png)

---

## 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [游戏操作](#游戏操作)
- [人脸替换使用指南](#人脸替换使用指南)
- [AI 精灵生成配置](#ai-精灵生成配置)
- [项目结构](#项目结构)
- [技术架构](#技术架构)
- [常见问题](#常见问题)
- [致谢](#致谢)

---

## 功能特性

- **AI 像素头部生成** - 通过 OpenAI GPT Image API 将照片转为 NES 风格像素画
- **完整精灵表替换** - AI 生成的头部自动合成到所有马里奥动画帧
- **调色板自动映射** - 一次生成，自动覆盖 Normal/Green/Red/Black/Fire 全部颜色变体
- **无敌闪烁正常工作** - 吃星星后颜色循环动画完全兼容
- **摄像头实时拍照** - 内嵌 Pygame 摄像头，带人脸引导框和实时检测
- **照片上传** - 支持从文件选择照片
- **手动选取人脸** - 自动检测失败时可手动框选
- **本地 Fallback** - 无 API Key 也能玩，自动降级为本地像素化
- **经典 Mario Level 1** - 完整还原第一关，包含所有敌人、道具和音效

---

## 快速开始

### 环境要求

- Python 3.10+
- 摄像头（可选，也可上传照片）

### 安装

```bash
# 克隆仓库
git clone https://github.com/thundecloud/Mario.git
cd Mario

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
cd Mario-Level-1
python mario_level_1.py
```

---

## 游戏操作

| 按键 | 功能 |
|------|------|
| `←` `→` | 左右移动 |
| `↑` `↓` | 上下（下蹲） |
| `A` | 跳跃 |
| `S` | 奔跑 / 发射火球 |
| `ESC` | 退出 |

### 游戏道具

| 道具 | 效果 |
|------|------|
| 蘑菇 | 小马里奥 → 大马里奥 |
| 火焰花 | 大马里奥 → 火焰马里奥（按 S 发射火球） |
| 星星 | 无敌状态（10 秒，颜色闪烁） |
| 金币 | +200 分，100 枚 = 1UP |
| 1UP 蘑菇 | 额外一条命 |

---

## 人脸替换使用指南

游戏启动后会先进入人脸替换界面，按以下流程操作：

### 第一步：获取人脸

启动游戏后，主菜单提供三个选项：

| 按钮 | 说明 |
|------|------|
| **Camera Capture** | 打开摄像头拍照，将脸对准椭圆引导框，按空格键拍摄 |
| **Upload Photo** | 从文件选择一张照片（支持 JPG/PNG） |
| **Skip** | 跳过人脸替换，直接玩原版马里奥 |

### 第二步：确认人脸

拍照/上传后进入预览界面：

- 左侧显示原始照片，右侧显示提取的人脸
- 点击 **Confirm** 确认并进入风格选择
- 点击 **Manual Select** 手动框选人脸区域（自动检测不准时使用）
- 点击 **Retake** 重新拍照

**手动选取**：在照片上拖拽画框选中人脸区域，点击 Confirm 确认。

### 第三步：选择风格

两种风格可选：

| 风格 | 说明 |
|------|------|
| **Sprite Art** (默认) | AI 生成像素风格头部 + 完整精灵表。若配置了 OpenAI API Key，会调用 AI 生成；否则本地像素化 |
| **Original** | 保持原始照片风格，直接贴到马里奥头部 |

- 点击风格卡片切换预览
- 选择 Sprite Art 后会显示 "Generating..." 加载动画（AI 生成需要几秒）
- 生成完成后点击 **Start Game!** 开始游戏

### 第四步：开始游戏

进入游戏后，马里奥的头部已被替换为你的像素人脸。以下动画状态全部生效：

- 站立、行走、跳跃、下蹲、死亡
- 小马里奥 ↔ 大马里奥 变身过渡
- 火焰马里奥
- 无敌闪烁（绿/红/黑色循环）
- 旗杆滑下

---

## AI 精灵生成配置

### 配置 OpenAI API Key

AI 像素头部生成需要 OpenAI API Key。无 Key 也可运行，系统会自动降级为本地像素化。

在项目根目录创建 `.env` 文件：

```
OPENAI_API_KEY=sk-proj-你的密钥
```

API Key 获取：[https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 生成流程说明

选择 Sprite Art 风格时，系统按以下流程生成：

```
用户照片
  → AI 生成 2 个像素头部（小马里奥 12x9px + 大马里奥 16x15px，4x 分辨率生成后缩小）
  → 颜色对齐到原始 NES 调色板
  → 逐帧合成（新头部 + 原始身体）写入 NORMAL 行
  → 从原始精灵表提取颜色映射
  → 自动生成 Green / Red / Black / Fire 变体行
  → 输出完整修改后的精灵表
  → 游戏中 mario.py 直接使用新精灵表，所有动画坐标不变
```

### Fallback 机制

| 条件 | 行为 |
|------|------|
| 有 API Key + 网络正常 | AI 生成像素头部，合成完整精灵表 |
| 无 API Key / 网络异常 | 本地像素化生成头部，合成完整精灵表 |
| 精灵表生成失败 | 回退到逐帧贴脸模式（旧方案） |
| 用户选择 Skip | 无替换，原版马里奥 |

---

## 项目结构

```
Mario/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── .env                         # OpenAI API Key 配置
└── Mario-Level-1/               # 游戏主目录
    ├── mario_level_1.py          # 启动入口
    ├── data/
    │   ├── main.py               # 游戏主函数 + face_swap_data 全局状态
    │   ├── setup.py              # Pygame 初始化 + 资源加载
    │   ├── constants.py          # 游戏常量（屏幕、物理、状态）
    │   ├── tools.py              # 工具函数
    │   ├── components/           # 游戏组件
    │   │   ├── mario.py          # 马里奥类（集成精灵表替换）
    │   │   ├── enemies.py        # 敌人（Goomba / Koopa）
    │   │   ├── powerups.py       # 道具（蘑菇 / 火焰花 / 星星）
    │   │   └── ...
    │   ├── states/               # 游戏状态
    │   │   ├── level1.py         # 第一关
    │   │   ├── main_menu.py      # 主菜单
    │   │   └── load_screen.py    # 加载 / 结束画面
    │   └── face_swap/            # 人脸替换模块
    │       ├── ui.py             # UI 界面（拍照/上传/风格选择）
    │       ├── face_capture.py   # 摄像头拍照 + 照片上传
    │       ├── face_detector.py  # MediaPipe 人脸检测
    │       ├── style_transfer.py # 风格转换引擎
    │       ├── ai_sprite.py      # OpenAI GPT Image AI 生成
    │       ├── sprite_replacer.py    # 逐帧头部替换（旧方案）
    │       ├── sprite_sheet_gen.py   # 完整精灵表生成（新方案）
    │       └── models/           # MediaPipe 模型文件（自动下载）
    └── resources/                # 游戏资源
        ├── graphics/             # 精灵表 + 关卡地图
        ├── music/                # 背景音乐
        ├── sound/                # 音效
        └── fonts/                # 字体
```

---

## 技术架构

### 依赖说明

| 库 | 用途 |
|------|------|
| **pygame** | 游戏引擎、摄像头画面渲染、UI 界面 |
| **opencv-python** | 摄像头采集、图像处理、颜色空间转换 |
| **mediapipe** | 人脸检测、面部特征点提取 |
| **numpy** | 图像数组操作、调色板映射 |
| **openai** | GPT Image API 调用（AI 像素画生成） |
| **python-dotenv** | 读取 .env 文件中的 API Key |
| **Pillow** | 图像格式处理 |

### 核心模块

| 模块 | 职责 |
|------|------|
| `SpriteSheetGenerator` | 生成完整精灵表：AI 头部 + 身体合成 + 调色板映射 |
| `PaletteMapper` | 从原始精灵表提取 NORMAL→变体的颜色映射关系 |
| `FaceSwapUI` | Pygame UI 状态机：main_menu → preview → style_select |
| `FaceDetector` | MediaPipe Tasks API 人脸检测封装 |
| `CameraCapture` | Pygame 内嵌摄像头 + 虚线椭圆引导 |
| `SpriteReplacer` | 逐帧头部替换（fallback 方案） |

### 精灵表替换原理

原始 `mario_bros.png` 精灵表中，各颜色变体的帧布局完全相同，仅颜色不同：

```
Y=0   : 大马里奥 NORMAL （11帧）
Y=48  : 大马里奥 FIRE   （11帧）
Y=144 : 大马里奥 BLACK  （8帧）
Y=192 : 大马里奥 GREEN  （8帧）
Y=240 : 大马里奥 RED    （8帧）

Y=32  : 小马里奥 NORMAL （11帧）
Y=176 : 小马里奥 BLACK  （6帧）
Y=224 : 小马里奥 GREEN  （6帧）
Y=272 : 小马里奥 RED    （6帧）
```

`SpriteSheetGenerator` 只需修改 NORMAL 行的头部，然后逐像素对比 NORMAL 和变体行提取颜色映射表，自动生成所有变体。`mario.py` 在初始化时将 `self.sprite_sheet` 替换为新 Surface，所有 `get_image(x, y, w, h)` 坐标不变，整个游戏无缝使用。

---

## 常见问题

### Q: 没有 OpenAI API Key 能玩吗？

可以。系统会自动降级为本地像素化处理，仍然会生成完整的精灵表。效果比 AI 生成略粗糙，但完全可玩。

### Q: 摄像头打不开怎么办？

- 确认摄像头未被其他应用占用
- 可以选择 **Upload Photo** 从文件上传照片代替

### Q: AI 生成一直 Loading 不动？

- 检查 `.env` 中的 API Key 是否正确
- 检查网络是否能访问 OpenAI API
- 等待约 10-15 秒是正常的，如果超过 30 秒可能是网络问题
- 生成失败后会自动降级为本地模式

### Q: 人脸检测不到怎么办？

- 确保照片中人脸清晰、正面、光线充足
- 使用 **Manual Select** 手动框选人脸区域

### Q: 无敌闪烁时颜色不对？

如果使用了精灵表生成模式（Sprite Art），颜色变体是通过调色板映射自动生成的，应该是正确的。如果颜色异常，可以尝试重新生成。

### Q: 如何恢复原版马里奥？

启动游戏后在人脸替换界面点击 **Skip (No Face Swap)** 即可。

---

## 致谢

- 游戏基座来自 [justinmeister/Mario-Level-1](https://github.com/justinmeister/Mario-Level-1)
- 人脸检测使用 [Google MediaPipe](https://developers.google.com/mediapipe)
- AI 像素画生成使用 [OpenAI GPT Image API](https://platform.openai.com/docs/guides/images)

---

**免责声明**：本项目仅用于非商业教育目的。Super Mario Bros 是任天堂的注册商标。
