# 第三方来源与许可

本仓库包含项目代码和若干第三方源码，不能将整个仓库理解为同一份授权。模型权重、MANO 数据、角色模型和视频没有随仓库分发。

| 组件 | 来源与说明 | 保留的许可 |
| --- | --- | --- |
| SAM 3D Body | Meta / facebookresearch 的本机源码快照；本地服务在 `local_app/` | `services/sam-3d-body/LICENSE`：SAM License |
| 公共 BF16 推理适配器 | 从 Comfy-Org/ComfyUI 选取并适配，来源提交记录见 vendor NOTICE | `services/sam-3d-body/local_app/vendor/COMFY_LICENSE`：GPL-3.0；Meta/MHR 源内声明保留 |
| SAM 2 | 本机使用的 SAM 2 源码与 YAML 配置 | `externals/sam2/LICENSE`：Apache-2.0；另见 `LICENSE_cctorch` |
| LivLiv 口型适配 | Python/NumPy 离线音频口型算法，注明上游提交和修改 | `tools/vendor/livliv/LICENSE`、`NOTICE`：Apache-2.0 |
| Wiggle 2 | 本机使用的 Blender 兼容版与渲染修正版 | `tools/vendor/wiggle2/LICENSE`：GPL-3.0；保留原 README、原版和本地版 |
| WiLoR | 本仓库保存接入代码与上游说明，WiLoR 本体及权重另行安装 | `third_party/wilor/license.txt`：CC-BY-NC-ND-4.0；MANO 与 Ultralytics 另有许可 |
| MMD Tools | Blender 运行依赖，由使用者单独安装 | 以上游发布的许可为准 |

本地服务 vendor NOTICE 中的 ComfyUI 上游提交为 `ba994879b88d0ebd9950b1e5c1e923d928f216ba`；LivLiv 来源提交为 `83ecd9c48f33d9439411e596841193c24dc7b005`。

模型下载公开可访问不改变其许可。项目自编部分本次仅作用户指定的源码归档，未新增统一开源许可证。
