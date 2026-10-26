"""Suite prompts must not encode the planted verdict."""

from __future__ import annotations

from collections import defaultdict

from peer_agent.evals import load_suite, review_prompt_text


def test_prereg_prompt_never_maps_one_text_to_multiple_truths():
    by_prompt: dict[str, set[str]] = defaultdict(set)
    for case in load_suite():
        by_prompt[review_prompt_text(case)].add(case.truth.name)
    collisions = {text: truths for text, truths in by_prompt.items() if len(truths) > 1}
    assert not collisions, f"prompt text leaks verdict: {list(collisions.items())[:3]}"


def test_prereg_prompt_omits_sample_size():
    text = review_prompt_text(load_suite()[0])
    assert "n_per_arm" not in text
    assert "per_arm" not in text.lower()
