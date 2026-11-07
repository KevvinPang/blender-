# Blender 动捕、表情与动作微调

本仓库保存 **2026-11-07 本机正在使用的整套代码**，包含视频动捕、本地识别服务、角色重定向、画面对齐、口型与表情时间轴，以及独立的动作微调插件。

| 功能 | 入口代码 | 本次版本 |
| --- | --- | --- |
| SAM 视频动捕、动作管理和修正 | `tools/sam_ellen_bridge.py` | 0.38.0；共享流程 revision 20 |
| 表情时间轴、分段表情、全局说话幅度 | `tools/ellen_expression_timeline.py` | 1.5.0 |
| 独立动作微调：选部位后 G / R | `tools/motion_tweak.py` | 0.3.2 |
| XR 输入、面部预设与 MMD 动作支持 | `tools/xr_ellen_bridge.py` | 0.3.13 |
| 头发等次级骨骼物理 | `tools/wiggle_2.py` | 当前本地修正版，保留上游许可 |

插件仍然分别注册，动作微调没有合并到 SAM 面板里。

## 包含哪些逻辑

- 本地 SAM 3D Body 视频识别、异步任务、取消、进度与缓存读取。
- WiLoR 手掌和手指接入、时序处理，以及 MediaPipe 面部、手部和可见手臂证据。
- MHR → MMD 的旋转重定向、手臂稳定、进出镜衔接、异常翻折保护、画面位置与远近对齐、出拳和持手机分段辅助。
- 原视频音频导入、配音口型、LivLiv 音频口型适配、说话幅度和表情叠加。
- 表情预设、按帧设置表情区间，以及对已有动捕的独立微调和原动作对照。

## 安装入口

详细步骤见 [安装说明](docs/INSTALL.md)，模块关系见 [代码结构](docs/ARCHITECTURE.md)。

Blender 端使用 **Blender 5.2.2**，并依赖已安装的 **MMD Tools**。识别任务使用独立 Python 环境。只需要编辑已有动作时，动作微调插件不需要启动识别服务。

```powershell
# 将插件和共享模块复制到明确指定的 Blender 插件目录
python scripts/install_blender_addons.py --addons-dir '你的 Blender scripts/addons 目录'

# 静态检查：源文件完整性、语法和项目模块依赖
python scripts/validate_sources.py

# 不启动 Blender 的仓库测试
python -m unittest discover -s tests -v
```

安装后重新启动 Blender，在偏好设置中启用所需插件；入口在 3D 视图的 N 侧栏。

## 代码与运行数据

仓库上传源代码、固定的手机握持参数、安装脚本、测试和第三方许可。角色 `.blend` / 模型、原视频、音频、推理结果、Python 虚拟环境、Blender 程序及模型权重保留在本机，按安装说明单独准备。

`docs/source_snapshot.json` 记录原始代码与仓库代码的 SHA-256。11 个文件仅调整部署路径，统一使用 `tools/mocap_paths.py`；本次整理没有修改动作求解、识别阈值或表情算法。原来的本机工程和插件未被仓库安装脚本自动更改。

## 当前边界

这是当前工作版本的源码归档，不代表所有视频的识别问题已经解决。画面对齐、遮挡、接触、快速出拳和手臂连续性仍需结合视频检查。动作微调采用固定骨长和关节约束，尚未实现完整的身体碰撞避让，也没有独立的动作速度重定时工具。

第三方代码和模型各有独立许可，见 [第三方说明](THIRD_PARTY.md)。本次归档没有为项目自编代码另行指定公开授权协议。
