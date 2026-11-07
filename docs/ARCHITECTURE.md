# 代码结构与数据流

## 本地识别

`services/sam-3d-body/local_app/app.py` 提供本机 HTTP 服务。`video_jobs.py` 管理识别任务，`runtime.py` 和 `public_backend.py` 连接 SAM 模型。面部、画面对齐和 WiLoR 由独立工作进程执行，避免 Python / CUDA 依赖冲突。

识别产物为 `SAM3D_BLENDER_TAKE` v1 JSON，包含时间、MHR 127 关节姿态及附加证据。原视频、缓存和模型不是源码的一部分。

## Blender 重定向与修正

入口 `tools/sam_ellen_bridge.py`，共享配置 `sam_ellen_pipeline.py`。`sam_ellen_retarget.py` 将关节旋转映射到角色骨架，并保持角色的休止骨长。

修正模块按当前生成流程组合使用，并非每个视频都会启用全部模块：

- `sam_ellen_screen*`、`image_*`、`camera_fit`、`projection`：画面对齐及投影拟合。
- `sam_ellen_arm_*`、`body_clearance`、`fold_guard`：整臂约束、证据判断、稳定与翻折保护。
- `sam_ellen_entry*`、`motion_continuity`、`temporal`：进出镜与连续性。
- `sam_ellen_punch*`、`fist_final`、`sam_punch_*`：出拳证据、拳头轮廓与姿态。
- `sam_ellen_phone*`：手机握持、手指与按帧区间。
- `sam_ellen_face`、`speech`、`audio_lipsync`、`livliv`：面部和说话口型。
- `sam_ellen_editing`、`sequence`、`video_audio`：片段管理、恢复与声音。

`enhance_sam_take_with_wilor.py`、`wilor_temporal.py`、`wilor_thumb.py` 和 `run_wilor_video_test.py` 构成 WiLoR 接入与后处理。`finger_reliability_repair.py` 为可选实验路径，并非无条件运行。

## 表情与动作微调

`ellen_expression_library.py` 提供形态键和预设，`xr_ellen_bridge.py` 提供共享面部控制，`ellen_expression_timeline.py` 管理分段表情、语音叠加和全局说话幅度。

`motion_tweak.py` 独立注册面板、点击选择、G / R 操作和非破坏性结果动作。它保留源 Action，以固定骨长解算修改结果，再写入正常动画曲线；渲染不依赖编辑器一直运行。

`wiggle_2.py` 保存本地使用的次级骨骼物理版本。原始副本与许可在 `tools/vendor/wiggle2/`。

## 快照完整性

`source_snapshot.json` 记录 216 个原始文件及其哈希。仓库包含 68 个生产模块和安装的物理插件源码。整理时仅在 11 个文件中替换运行路径；其余快照文件与当前本机源文件内容一致。

新增 `mocap_paths.py`、服务侧 `project_paths.py`、安装脚本和仓库测试，不改生成算法。校验脚本检查原始文件哈希、仓库文件哈希以及本地模块是否齐全。
