import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from scripts.model_lifecycle_manager import ModelLifecycleManager, ModelSpec, Residency
from scripts.model_registry import (
    build_registry,
    GEOCHAT,
    QWEN_CDVQA,
    GENERAL_VLM,
)
from scripts.geochat_specialist import GeoChatSpecialist


def test_geochat_unload_signature():
    print("=== Testing GeoChatSpecialist.unload signature ===")
    spec = GeoChatSpecialist()
    # Model not loaded yet, call unload with force_release=True
    try:
        spec.unload(force_release=True)
        print("Calling spec.unload(force_release=True) succeeded with no TypeError.")
    except TypeError as e:
        raise AssertionError(f"spec.unload(force_release=True) raised TypeError: {e}")

    try:
        spec.unload(force_release=False)
        print("Calling spec.unload(force_release=False) succeeded.")
    except TypeError as e:
        raise AssertionError(f"spec.unload(force_release=False) raised TypeError: {e}")

    try:
        spec.unload()
        print("Calling spec.unload() succeeded.")
    except TypeError as e:
        raise AssertionError(f"spec.unload() raised TypeError: {e}")


def test_lifecycle_manager_swap_isolation():
    print("\n=== Testing Lifecycle Manager Swap & Slot Isolation ===")
    manager = build_registry(device="cuda" if torch.cuda.is_available() else "cpu")

    # Confirm GEOCHAT spec has shared_group == None
    geochat_spec = manager._specs[GEOCHAT]
    qwen_spec = manager._specs[QWEN_CDVQA]
    general_vlm_spec = manager._specs[GENERAL_VLM]

    print(f"GEOCHAT shared_group: {geochat_spec.shared_group}")
    print(f"QWEN_CDVQA shared_group: {qwen_spec.shared_group}")
    print(f"GENERAL_VLM shared_group: {general_vlm_spec.shared_group}")

    assert geochat_spec.shared_group is None, (
        f"Expected GEOCHAT shared_group to be None, got {geochat_spec.shared_group}"
    )
    assert qwen_spec.shared_group == "qwen_vl", "Expected QWEN_CDVQA to have shared_group='qwen_vl'"
    assert general_vlm_spec.shared_group == "qwen_vl", "Expected GENERAL_VLM to have shared_group='qwen_vl'"

    # Now verify swap eviction behavior using mock specialists to trace calls and memory state
    unloaded_models = []

    class MockSpecialist:
        def __init__(self, name):
            self.name = name
            self.loaded = True

        def unload(self, force_release: bool = False):
            unloaded_models.append((self.name, force_release))
            self.loaded = False

    def mock_unload(s):
        s.unload(force_release=True)

    test_mgr = ModelLifecycleManager(device="cpu")
    test_mgr.register(ModelSpec(
        name=QWEN_CDVQA,
        loader_fn=lambda: MockSpecialist(QWEN_CDVQA),
        residency=Residency.SWAPPABLE,
        unloader_fn=mock_unload,
        shared_group="qwen_vl",
    ))
    test_mgr.register(ModelSpec(
        name=GEOCHAT,
        loader_fn=lambda: MockSpecialist(GEOCHAT),
        residency=Residency.SWAPPABLE,
        unloader_fn=mock_unload,
        shared_group=None,  # FIXED: isolated slot
    ))

    # Step 1: Load Qwen-based specialist
    print("Step 1: Loading Qwen specialist...")
    qwen_obj = test_mgr.load(QWEN_CDVQA)
    assert test_mgr._swap_slot_occupant == QWEN_CDVQA
    assert QWEN_CDVQA in test_mgr._loaded
    print(f"Slot occupant: {test_mgr._swap_slot_occupant}, Loaded models: {list(test_mgr._loaded.keys())}")

    # Step 2: Load GeoChat
    print("Step 2: Loading GeoChat specialist (should evict Qwen)...")
    geochat_obj = test_mgr.load(GEOCHAT)
    print(f"Slot occupant: {test_mgr._swap_slot_occupant}, Loaded models: {list(test_mgr._loaded.keys())}")

    # Assert Qwen was evicted
    assert test_mgr._swap_slot_occupant == GEOCHAT
    assert GEOCHAT in test_mgr._loaded
    assert QWEN_CDVQA not in test_mgr._loaded, "Error: QWEN_CDVQA is still resident in manager after swapping to GeoChat!"
    assert (QWEN_CDVQA, True) in unloaded_models, f"Error: Qwen unload(force_release=True) was not invoked! Unload log: {unloaded_models}"
    print(f"Eviction verified: Qwen was unloaded with force_release=True. Unload log: {unloaded_models}")


if __name__ == "__main__":
    test_geochat_unload_signature()
    test_lifecycle_manager_swap_isolation()
    print("\nALL FIX 3 LIFECYCLE CHECKS PASSED.")
