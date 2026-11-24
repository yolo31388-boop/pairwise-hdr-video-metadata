import pytest
import media.hdr_video_metadata as mod

CASES = [
    ("per_frame_meta", (1,)),
    ("transfer_branch", (1,)),
    ("smooth_params", (1,)),
    ("frame_meta_bind", (1,)),
    ("reset_on_format", (1,)),
    ("gamut_clip", (1,)),
    ("probe", (1,)),
    ("decode", (1,)),
    ("map", (1,)),
    ("present", (1,)),
    ("validate_luma", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
