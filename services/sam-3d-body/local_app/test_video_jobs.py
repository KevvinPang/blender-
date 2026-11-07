import json
import math

import numpy as np
import pytest

from video_jobs import OUTPUT_TO_BLENDER, RAW_TO_BLENDER, get_job, sample_indices


def test_sample_times_and_duration():
    values = sample_indices(554, 30, 10)
    assert len(values) == 186
    assert values[0] == 0 and values[-1] == 553
    assert values == sorted(set(values))
    assert sample_indices(60, 30, 30) == list(range(60))
    trimmed = sample_indices(554, 30, 10, 2, 3)
    assert trimmed[0] == 60 and trimmed[-1] == 89
    assert len(sample_indices(30, 10, 30)) == 30


@pytest.mark.parametrize('args', [(0, 30, 10), (100, 0, 10), (100, 30, 0),
                                (100, 30, 10, 4, 3), (100, 30, 10, math.nan, 0)])
def test_invalid_intervals(args):
    with pytest.raises(ValueError):
        sample_indices(*args)


def test_coordinate_conversion():
    np.testing.assert_allclose(RAW_TO_BLENDER @ np.array([1, 2, 3]), [1, -3, 2])
    np.testing.assert_allclose(OUTPUT_TO_BLENDER @ np.array([1, -2, -3]), [1, -3, 2])
    assert np.linalg.det(RAW_TO_BLENDER) == 1
    np.testing.assert_allclose(RAW_TO_BLENDER @ RAW_TO_BLENDER.T, np.eye(3))


def test_restarted_job_and_path_validation(tmp_path, monkeypatch):
    monkeypatch.setattr('video_jobs.JOBS_DIR', tmp_path)
    job_id = '0' * 32
    (tmp_path/job_id).mkdir()
    (tmp_path/job_id/'status.json').write_text(json.dumps({'state': 'running'}))
    assert get_job(job_id)['state'] == 'error'
    with pytest.raises(ValueError):
        get_job('../../file')
