import pytest

from slime.observability.trace_utils import TRACE_CHILDREN_KEY, build_sglang_meta_trace_attrs

NUM_GPUS = 0


@pytest.mark.unit
def test_build_sglang_meta_trace_attrs_keeps_standard_and_pd_fields():
    meta = {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "cached_tokens": 3,
        "pd_prefill_forward_duration": 0.125,
        "pd_decode_transfer_duration": 0.05,
        "finish_reason": {"type": "stop"},
        "unused_field": "ignored",
    }

    attrs = build_sglang_meta_trace_attrs(meta)
    trace_children = attrs.pop(TRACE_CHILDREN_KEY)

    assert attrs == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "cached_tokens": 3,
        "finish_reason": "stop",
    }
    assert trace_children[0]["name"] == "sglang_pd_prefill"
    assert trace_children[0]["children"][0]["attrs"] == {
        "pd_prefill_forward_duration": 0.125,
    }
    assert trace_children[1]["name"] == "sglang_pd_decode"
    assert trace_children[1]["children"][0]["attrs"] == {
        "pd_decode_transfer_duration": 0.05,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
