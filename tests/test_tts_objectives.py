"""WER semantics and real policy-gradient directions for the TTS objectives."""

import pytest
import torch

from slime.rollout.rm_hub.wer import word_error_rate
from slime.utils.ppo_utils import action_policy_terms, mopd_action_terms

NUM_GPUS = 0


@pytest.mark.parametrize(
    "reference,hypothesis,expected",
    [
        ("Hello, WORLD!", "hello world", 0),
        ("one two three", "one three", 1 / 3),
        ("one two", "one four", 0.5),
        ("one", "one two three", 2),
        ("one two", "", 1),
    ],
)
def test_wer_preserves_insertions_and_empty_hypotheses(reference, hypothesis, expected):
    assert word_error_rate(reference, hypothesis)["wer"] == pytest.approx(expected)


def test_empty_normalized_reference_matches_delay_definition():
    # The requested Delay definition preserves raw +inf and clips reward to zero.
    assert word_error_rate("...", "hello")["wer"] == float("inf")
    assert 1 - min(word_error_rate("...", "hello")["wer"], 1) == 0


def test_grpo_positive_and_negative_advantages_move_selected_probability():
    logits = torch.zeros((2, 2), requires_grad=True)
    selected = logits.log_softmax(-1)[:, 0]
    old = selected.detach().clone()
    terms, ratios, _ = action_policy_terms(selected, old, torch.tensor([1.0, -1.0]), torch.ones(2, dtype=torch.bool))
    terms.sum().backward()
    assert logits.grad[0, 0] < 0  # descent increases a positively rewarded action
    assert logits.grad[1, 0] > 0  # descent decreases a negatively rewarded action
    torch.testing.assert_close(ratios, torch.ones(2))


def test_grpo_old_policy_stays_fixed_and_clipping_has_effect():
    current = torch.tensor([-0.1], requires_grad=True)
    old = torch.tensor([-1.0])
    terms, ratio, clipped = action_policy_terms(current, old, torch.ones(1), torch.ones(1, dtype=torch.bool))
    terms.sum().backward()
    assert ratio.item() > 2
    assert clipped.item() == 1
    assert current.grad.item() == 0


def test_equal_teacher_mopd_has_zero_gradient():
    current = torch.tensor([-1.0, -2.0], requires_grad=True)
    old = current.detach().clone()
    teacher = old.clone().requires_grad_()
    terms, _ = mopd_action_terms(current, old, teacher, torch.ones(2, dtype=torch.bool))
    terms.sum().backward()
    assert current.grad.eq(0).all()
    assert teacher.grad is None


def test_masked_positions_do_not_poison_losses():
    current = torch.tensor([-0.3, float("nan")], requires_grad=True)
    old = torch.tensor([-0.3, float("nan")])
    teacher = torch.tensor([-0.1, float("nan")])
    mask = torch.tensor([True, False])
    terms, _ = mopd_action_terms(current, old, teacher, mask)
    assert torch.isfinite(terms).all()
    terms.sum().backward()
    assert current.grad[1] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
