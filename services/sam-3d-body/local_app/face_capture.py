"""Run facial inference outside the SAM/Torch environment; allow cancellation."""
from project_paths import MOTION_PYTHON
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name('face_worker.py')
PYTHON = MOTION_PYTHON


class FaceCancelled(Exception):
    pass


def run_face(video, folder, rate, start, end, cancel, progress):
    if not PYTHON.is_file():
        raise RuntimeError('The separate facial inference environment was not found')
    output = Path(folder) / 'face_take.json'
    command = [str(PYTHON), str(WORKER), '--video', str(video), '--output', str(output),
               '--rate', str(rate), '--start', str(start), '--end', str(end)]
    with (Path(folder)/'face_worker.log').open('w', encoding='utf8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while process.poll() is None:
                if cancel.is_set():
                    raise FaceCancelled()
                status = Path(folder)/'face_progress.json'
                if status.is_file():
                    try:
                        progress(**json.loads(status.read_text(encoding='utf8')))
                    except (ValueError, OSError):
                        pass
                time.sleep(.1)
            if process.returncode:
                raise RuntimeError('Facial capture failed; see '+str(Path(folder)/'face_worker.log'))
            if cancel.is_set():
                raise FaceCancelled()
            return json.loads(output.read_text(encoding='utf8'))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def run_alignment(payload, folder, cancel, progress):
    folder = Path(folder)
    source, output = folder/'alignment_body.json', folder/'alignment.json'
    source.write_text(json.dumps(payload, allow_nan=False), encoding='utf8')
    command = [str(PYTHON), str(WORKER.with_name('alignment_worker.py')),
               '--body', str(source), '--output', str(output)]
    with (folder/'alignment_worker.log').open('w', encoding='utf8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while process.poll() is None:
                if cancel.is_set():
                    raise FaceCancelled()
                status = folder/'alignment_progress.json'
                if status.is_file():
                    try:
                        progress(**json.loads(status.read_text(encoding='utf8')))
                    except (ValueError, OSError):
                        pass
                time.sleep(.1)
            if cancel.is_set():
                raise FaceCancelled()
            if process.returncode:
                raise RuntimeError('Hand screen detection failed; see '+str(folder/'alignment_worker.log'))
            return json.loads(output.read_text(encoding='utf8'))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
