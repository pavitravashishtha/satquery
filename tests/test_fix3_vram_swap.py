import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_registry import build_registry, GEOCHAT, QWEN_CDVQA
from scripts.geochat_specialist import GeoChatSpecialist


def test_vram_swap():
    print("=== Testing Fix 3: VRAM Swap and Unload Behavior ===")

    manager = build_registry(device="cuda")

    # Initial VRAM
    torch.cuda.empty_cache()
    vram_start_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Initial VRAM allocated: {vram_start_gb:.3f} GB")

    # 1. Load Qwen-based specialist (ChangeVQA)
    print("\n1. Loading QWEN_CDVQA specialist...")
    qwen_specialist = manager.load(QWEN_CDVQA)
    vram_qwen_gb = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after loading QWEN_CDVQA: {vram_qwen_gb:.3f} GB")
    assert QWEN_CDVQA in manager._loaded, "QWEN_CDVQA should be in manager._loaded"
    assert manager._swap_slot_occupant == QWEN_CDVQA

    # 2. Load GeoChat specialist (should evict Qwen from the swap slot)
    print("\n2. Loading GEOCHAT specialist (swapping out QWEN_CDVQA)...")
    geochat_specialist = manager.load(GEOCHAT)
    vram_geochat_gb = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after loading GEOCHAT: {vram_geochat_gb:.3f} GB")

    # Assert Qwen weights are NO LONGER resident
    assert QWEN_CDVQA not in manager._loaded, (
        f"Assertion Failed: QWEN_CDVQA is still resident in manager._loaded: {manager._loaded.keys()}"
    )
    assert GEOCHAT in manager._loaded, "GEOCHAT should be in manager._loaded"
    assert manager._swap_slot_occupant == GEOCHAT, f"Slot occupant should be GEOCHAT, got {manager._swap_slot_occupant}"
    print("Assertion passed: Qwen specialist was cleanly evicted from manager._loaded.")

    # Confirm no co-existence: on this 6GB GPU (RTX 4050 Laptop), coexistence of Qwen (2.4GB) + GeoChat (4.2GB)
    # would equal 6.6GB and trigger a CUDA OOM. Allocating ~4.2GB proves Qwen was freed.
    assert vram_geochat_gb < 5.5, f"VRAM ({vram_geochat_gb:.2f} GB) indicates potential model coexistence!"
    print(f"VRAM check passed: {vram_geochat_gb:.2f} GB < 5.5 GB (no Qwen/GeoChat coexistence)")

    # 3. Direct specialist.unload(force_release=True) verification
    print("\n3. Testing specialist.unload(force_release=True) signature...")
    try:
        geochat_specialist.unload(force_release=True)
        print("Success: geochat_specialist.unload(force_release=True) executed without TypeError.")
    except TypeError as e:
        raise AssertionError(f"TypeError raised on unload(force_release=True): {e}")

    # Manager cleanup
    manager.unload(GEOCHAT)
    torch.cuda.empty_cache()
    vram_final_gb = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after final unload: {vram_final_gb:.3f} GB")

    print("\nALL FIX 3 VERIFICATIONS PASSED.")


if __name__ == "__main__":
    test_vram_swap()
