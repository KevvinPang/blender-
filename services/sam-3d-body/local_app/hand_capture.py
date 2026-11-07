"""Run WiLoR in its isolated environment after SAM releases GPU memory."""
from project_paths import TOOLS, WILOR_PYTHON
import json
import subprocess
import time
from pathlib import Path

from face_capture import FaceCancelled


PYTHON = WILOR_PYTHON


def available():
    root = PYTHON.parents[2]
    return (PYTHON.is_file() and (TOOLS/'enhance_sam_take_with_wilor.py').is_file()
            and all((root/name).is_file() for name in (
                'pretrained_models/wilor_final.ckpt', 'pretrained_models/detector.pt',
                'pretrained_models/model_config.yaml', 'mano_data/MANO_RIGHT.pkl')))


def run_hands(payload, folder, side, cancel, progress):
    if not available():
        raise RuntimeError('WiLoR environment or model weights not found')
    folder = Path(folder)
    source, output = folder/'hands_body.json', folder/'hands_take.json'
    source.write_text(json.dumps(payload, allow_nan=False), encoding='utf8')
    command = [str(PYTHON), str(TOOLS/'enhance_sam_take_with_wilor.py'),
               '--take', str(source), '--output', str(output), '--side', side,
               '--progress', str(folder/'hands_progress.json')]
    with (folder/'hands_worker.log').open('w', encoding='utf8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while process.poll() is None:
                if cancel.is_set():
                    raise FaceCancelled()
                status = folder/'hands_progress.json'
                if status.is_file():
                    try:
                        progress(**json.loads(status.read_text(encoding='utf8')))
                    except (ValueError, OSError):
                        pass
                time.sleep(.1)
            if cancel.is_set():
                raise FaceCancelled()
            if process.returncode:
                raise RuntimeError('WiLoR failed; see '+str(folder/'hands_worker.log'))
            return json.loads(output.read_text(encoding='utf8'))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
