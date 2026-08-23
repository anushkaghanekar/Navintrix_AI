"""Load-bearing tests for the core scope. These are the ones that matter
most — if only a few tests get written, write these.

Fill in each with pytest.skip("TODO") removed once the corresponding
module exists.
"""

import pytest


def test_counting_does_not_double_count_same_track_id():
    pytest.skip("TODO: implement once counting/counter.py exists")


def test_state_machine_never_skips_yellow_and_all_red():
    pytest.skip("TODO: implement once controller/state_machine.py exists")


def test_state_machine_refuses_conflicting_green():
    pytest.skip("TODO: implement once controller/state_machine.py exists")


def test_state_machine_enforces_min_green():
    pytest.skip("TODO")


def test_state_machine_enforces_max_green():
    pytest.skip("TODO")


def test_emergency_priority_only_triggers_when_actually_approaching():
    pytest.skip("TODO: emergency/detector.py's is_approaching_intersection")


def test_emergency_priority_still_goes_through_safety_transition():
    pytest.skip("TODO: controller/emergency_controller.py + state_machine.py")


def test_fairness_forces_green_after_max_wait():
    pytest.skip("TODO: controller/fairness.py")


def test_dataset_split_has_no_video_leakage_across_train_test():
    pytest.skip("TODO: scripts/split_dataset.py")
