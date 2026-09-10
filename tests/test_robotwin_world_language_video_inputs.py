import copy
import pytest
from scripts.audit_robotwin_world_language_video_inputs import loss_inputs


def rows():
    return [dict(reference=r, sigma=1., seed=42, noisy_sha256='a' * 64,
                 errors={'source': {'mse': 1., 'object_roi_mse': 2.},
                         'target': {'mse': 3., 'object_roi_mse': 5.}},
                 wrong_minus_correct_mse=2. if r == 'source' else -2.,
                 wrong_minus_correct_roi_mse=3. if r == 'source' else -3.)
            for r in ['source', 'target']]


def test_margin_direction_tracks_reference_and_pure_noise_matches():
    result = loss_inputs(rows(), ['source', 'target'], [1.], [42])
    assert result['source', 1., 42] == result['target', 1., 42]


@pytest.mark.parametrize('corruption', ['margin', 'noise', 'missing', 'duplicate'])
def test_rejects_mispaired_or_corrupted_cells(corruption):
    data = copy.deepcopy(rows())
    if corruption == 'margin': data[1]['wrong_minus_correct_roi_mse'] = 3.
    if corruption == 'noise': data[1]['noisy_sha256'] = 'b' * 64
    if corruption == 'missing': data.pop()
    if corruption == 'duplicate': data.append(data[0])
    with pytest.raises(ValueError): loss_inputs(data, ['source', 'target'], [1.], [42])
