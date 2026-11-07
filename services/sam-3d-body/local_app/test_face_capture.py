import json
import threading
import time

import pytest

import video_jobs as jobs
from face_capture import FaceCancelled


@pytest.mark.parametrize('value', [0, 61, float('nan'), '30'])
def test_face_rate_rejected_before_work(value):
    with pytest.raises(ValueError):
        jobs.start_face_job('0'*32, value)


def test_face_only_uses_original_body_and_supports_cancellation(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'JOBS_DIR', tmp_path)
    monkeypatch.setattr(jobs, '_jobs', {})
    monkeypatch.setattr(jobs, '_cancels', {})
    monkeypatch.setattr(jobs.runtime, 'unload_model', lambda: None)
    source_id = '0'*32
    folder = tmp_path/source_id
    folder.mkdir()
    video = tmp_path/'source.mp4'
    video.touch()
    body = {'job_id': source_id, 'video': str(video), 'fps': 30,
            'frames': [{'source_frame': 30, 'time': 1, 'positions': ['unchanged']},
                       {'source_frame': 60, 'time': 2, 'positions': ['unchanged']}],
            'limitations': ['single-person', 'no facial tracking']}
    original = json.dumps(body)
    (folder/'take.json').write_text(original)
    (folder/'status.json').write_text(json.dumps({'id': source_id, 'state': 'complete', 'fps': 30}))

    def fake_face(video, folder, rate, start, end, cancel, progress):
        assert start == 1 and end == 61/30 and rate == 30
        progress(done=2, total=2)
        return {'limitations': ['visual-only'], 'frames': [1, 2]}

    monkeypatch.setattr(jobs, 'run_face', fake_face)
    job = jobs.start_face_job(source_id)

    def finished(job):
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            result = jobs.get_job(job['id'])
            if result['state'] in ('complete', 'error', 'cancelled'):
                return result
            time.sleep(.01)
        raise AssertionError('Job timed out')

    status = finished(job)
    assert status['state'] == 'complete', status
    combined = json.loads((tmp_path/job['id']/'take.json').read_text())
    assert combined['frames'] == body['frames']
    assert combined['limitations'] == ['single-person', 'visual-only']
    assert (folder/'take.json').read_text() == original

    entered = threading.Event()
    def cancellable(*args):
        entered.set()
        if args[-2].wait(3):
            raise FaceCancelled()
        raise AssertionError('Cancel was not delivered')
    monkeypatch.setattr(jobs, 'run_face', cancellable)
    job = jobs.start_face_job(source_id)
    assert entered.wait(2)
    jobs.cancel_job(job['id'])
    assert finished(job)['state'] == 'cancelled'
    assert not (tmp_path/job['id']/'take.json').exists()
