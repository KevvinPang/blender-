# 安装与运行

## 1. 运行环境

- Windows 10/11；Blender 5.2.2。
- Blender 中安装并启用 MMD Tools；使用带 MMD 骨骼和面部形态键的角色工程。
- 识别服务和手部推理使用 NVIDIA CUDA；具体显存、速度取决于模型与输入视频。
- 系统 PATH 中可找到 `ffmpeg` 和 `ffprobe`，用于音频读取、视频声音导入及配音口型。

目前这套代码按艾莲、玲的骨架与形态键适配；动作微调识别 `Arm_L/R`、`Elbow_L/R`、`Wrist_L/R` 以及对应手指名称。其他角色需要确认骨架映射。

## 2. 安装 Blender 插件

```powershell
python scripts/install_blender_addons.py --addons-dir 'C:/你的Blender/portable/scripts/addons'
```

此脚本复制所有运行模块、手机握持 JSON，并写入本机 `mocap_runtime.json` 指向当前仓库。覆盖已有同名文件前会在目标目录的 `mocap_install_backups` 下备份。不要把仓库移动或删除后继续使用旧的路径配置；移动后重新运行安装脚本即可。

重新启动 Blender，在偏好设置中启用：

1. MMD Tools。
2. Wiggle 2（需要头发物理时）。
3. XR Animator - Ellen Live Preview（表情共享依赖）。
4. SAM 3D Body - Ellen Preview。
5. Ellen Expression Timeline。
6. 动作微调 / Motion Tweak。

也可以明确通过 Blender 执行 `scripts/enable_blender_addons.py` 启用第 2—6 项并保存偏好设置。该脚本不安装 MMD Tools，也不打开或保存角色工程。

## 3. 面部、画面对齐工作环境

建议使用 Python 3.11，避免与 Blender 或 SAM 的 PyTorch 环境混装：

```powershell
py -3.11 -m venv .venvs/motion
.venvs/motion/Scripts/python.exe -m pip install -r requirements-motion.txt
```

将以下 MediaPipe 官方模型放入 `services/sam-3d-body/local_app/models/`：

- `face_landmarker.task`
- `hand_landmarker.task`

模型按各自官方许可获取。代码保留面部模型 SHA-256 校验。可见前臂证据还使用 MediaPipe 0.10.21 自带的传统 Pose 图，不能随意升级到删除 `mp.solutions` 的版本。

## 4. SAM 服务与模型

```powershell
py -3.11 -m venv services/sam-3d-body/.venv
services/sam-3d-body/.venv/Scripts/python.exe -m pip install -r requirements-service.txt
```

当前 PyTorch 依赖为 CUDA 12.8 版。使用仓库中的独立公共 BF16 适配器时，将 `sam_3d_body_dinov3_bf16.safetensors` 放在 `models/sam3d/`，或用环境变量 `SAM3D_WEIGHTS` 指定位置。

公共权重来源：<https://huggingface.co/Comfy-Org/sam-3d-body>。实际代码会检查文件大小和 SHA-256；名称相同但内容不同的权重不会被当作已验证模型。公共后端的头部和骨架使用 float32，骨干使用 BF16；并不承诺与原始 Meta 后端逐位相同。

原始 Meta checkpoint 是另一种后端：按 `services/sam-3d-body/INSTALL.md` 获取授权后，将 `model.ckpt`、`model_config.yaml` 和 `assets/mhr_model.pt` 放入 `services/sam-3d-body/checkpoints/sam-3d-body-dinov3/`。

启动：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_service.ps1
```

默认地址 `http://127.0.0.1:7872`。Blender 的 SAM 面板中填写同一地址。服务只接受本机使用；不要向公网开放。生成任务、缓存和日志写入服务目录的 `local_output/`，均由 Git 忽略。

## 5. WiLoR 手部推理

项目的 WiLoR 接入、手指修正与时序处理代码已在 `tools/` 保存。WiLoR 本体按上游方式安装到独立目录：

```powershell
git clone --recursive https://github.com/rolpotamias/WiLoR.git externals/WiLoR
```

按其 README 创建 `externals/WiLoR/.venv`、安装依赖并准备：

- `pretrained_models/wilor_final.ckpt`
- `pretrained_models/detector.pt`
- `pretrained_models/model_config.yaml`
- `mano_data/MANO_RIGHT.pkl` 与上游要求的其他 MANO 数据。

WiLoR、YOLO 和 MANO 各自许可不同，参见 `third_party/wilor/` 和上游说明。权重、MANO 数据及 WiLoR 外部 checkout 不提交到本仓库。

需要根据视频追踪近镜头拳头轮廓时，还需要在 WiLoR 环境安装仓库附带的 SAM 2 源码，并准备 SAM 2.1 tiny 权重：

```powershell
externals/WiLoR/.venv/Scripts/python.exe -m pip install -e externals/sam2
```

权重放到 `models/sam2/sam2.1_hiera_tiny.pt`，或设置 `SAM2_CHECKPOINT`。Windows 上 SAM 2 的 CUDA 扩展需按其上游安装说明处理。

## 6. 自定义路径

`tools/mocap_paths.py` 优先读取进程环境变量，再使用仓库相对路径。`.env.example` 是参考文件，不会被自动加载。

| 变量 | 用途 |
| --- | --- |
| `MOCAP_ROOT` | 整个仓库根目录 |
| `SAM3D_ROOT` | SAM 服务及上游源码目录 |
| `MOCAP_MOTION_PYTHON` | 面部和画面对齐工作进程的 Python |
| `WILOR_ROOT` | WiLoR 源码及模型目录 |
| `MOCAP_WILOR_PYTHON` | WiLoR 工作进程的 Python |
| `SAM3D_WEIGHTS` | 公共 BF16 权重路径 |
| `SAM2_ROOT` | SAM 2 源码路径 |
| `SAM2_CHECKPOINT` | SAM 2.1 tiny 权重路径 |

例如：

```powershell
$env:MOCAP_MOTION_PYTHON = 'D:/python/motion/Scripts/python.exe'
$env:WILOR_ROOT = 'D:/models/WiLoR'
$env:MOCAP_WILOR_PYTHON = 'D:/models/WiLoR/.venv/Scripts/python.exe'
```

Blender 需要在这些环境变量设置后启动。配置也保留原本桌面/下载目录的兼容回退，便于本机迁移。

## 7. 独立动作微调

已有动捕时，进入 N 侧栏“动作微调”，选择骨架并建立微调层。支持双臂、手掌与手指。

- **修附近**：暂停在要改的帧，点击部位后 G / R，确认后自动记录，默认前后各 8 帧平滑过渡。
- **修整段**：设置起止帧，G / R 调整后点击“应用到整段”。起止帧也可编辑。
- 源动作与微调结果独立保存，可对照；重新生成动捕时按当前插件的动作绑定规则处理。
- 结果是普通动画曲线，可保存并用于原生 Ctrl+F12 渲染。

插件暂未提供独立速度重定时工具。修拳头的位置不等于同时修好了出拳速度，需要播放核对。

## 8. 检查范围

```powershell
python scripts/validate_sources.py
python -m unittest discover -s tests -v
```

Blender 注册与卸载检查：

```powershell
blender --background --factory-startup --python scripts/smoke_blender.py
```

服务单元测试需要在服务环境安装 pytest 后执行 `python -m pytest services/sam-3d-body/local_app`。其中某些历史测试依赖本机模型或环境，测试通过不代替模型推理与角色工程的视觉检查。

`services/sam-3d-body/local_app/DEPLOYMENT.md` 是原本早期部署记录，可能描述旧功能范围；当前功能以本仓库 README 和源码为准。
