import argparse
import json
import traceback

from runtime import MODEL_URL, OUTPUT_DIR, diagnostics, download_model, infer_image, missing_weights, read_video_frame, unload_model


def build_app():
    import gradio as gr

    def run(image, metadata):
        try:
            return infer_image(image, metadata)
        except Exception as error:
            traceback.print_exc()
            raise gr.Error(str(error)) from None

    def extract(path, seconds):
        try:
            return read_video_frame(path, seconds)
        except Exception as error:
            raise gr.Error(str(error)) from None

    def download():
        try:
            download_model()
            return diagnostics()
        except Exception as error:
            raise gr.Error(str(error)) from None

    with gr.Blocks(title="SAM 3D Body Local", analytics_enabled=False) as app:
        with gr.Tabs():
            with gr.Tab("识别测试"):
                source = gr.State(None)
                with gr.Row():
                    with gr.Column():
                        video = gr.File(label="视频文件", file_types=["video"], type="filepath")
                        seconds = gr.Number(label="视频时间（秒）", value=0, minimum=0)
                        extract_button = gr.Button("提取视频帧")
                        image = gr.Image(label="输入图片 / 视频帧", type="numpy", image_mode="RGB", height=400)
                        run_button = gr.Button("识别这一帧", variant="primary")
                    with gr.Column():
                        overlay = gr.Image(label="身体与手指关键点", height=400)
                        plot = gr.Plot(label="三维骨架")
                data = gr.File(label="原始预测数据（NPZ）")
                report = gr.JSON(label="本次识别记录")
                extract_button.click(extract, inputs=[video, seconds], outputs=[image, source])
                image.upload(lambda: None, outputs=source)
                image.clear(lambda: None, outputs=source)
                run_button.click(run, inputs=[image, source], outputs=[overlay, plot, data, report], concurrency_id="model", concurrency_limit=1)
            with gr.Tab("环境与模型"):
                state = gr.JSON(label="环境状态", value=diagnostics())
                with gr.Row():
                    refresh = gr.Button("刷新状态")
                    download_button = gr.Button("验证本地模型 / 补全官方权重")
                    unload = gr.Button("卸载模型 / 释放显存")
                gr.Button("官方模型授权页", link=MODEL_URL)
                refresh.click(diagnostics, outputs=state)
                download_button.click(download, outputs=state, concurrency_id="model", concurrency_limit=1)
                unload.click(unload_model, outputs=state, concurrency_id="model", concurrency_limit=1)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, default=7872)
    args = parser.parse_args()
    if args.check:
        result = diagnostics(check_imports=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        failed = any(value != "OK" for key, value in result.items() if key.startswith("import_"))
        raise SystemExit(1 if failed or result.get("cuda_matmul") is not True else 0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI

    api = FastAPI()
    from video_jobs import mount_api
    mount_api(api)

    @api.get("/sam3d-health")
    def health():
        return {"app": "sam-3d-body-local", "weights_ready": not missing_weights(), "blender_bridge": 1,
                "video_display_orientation": True, "face_capture": 1, "screen_alignment": 1,
                "wilor_hands": 1, "sequential_video_decode": 1}

    api = gr.mount_gradio_app(api, build_app().queue(max_size=8), path="/", allowed_paths=[str(OUTPUT_DIR)])
    uvicorn.run(api, host="127.0.0.1", port=args.port, access_log=False)
