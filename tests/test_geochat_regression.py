import os
import sys
import gc
import torch
import torch.nn as nn

# Add satquery and GeoChat to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GEOCHAT_ROOT = os.path.join(PROJECT_ROOT, "GeoChat")
if GEOCHAT_ROOT not in sys.path:
    sys.path.insert(0, GEOCHAT_ROOT)

from scripts.model_registry import build_registry, GEOCHAT, _load_geochat
from scripts.geochat_specialist import GeoChatSpecialist
from scripts.general_vlm_specialist import GeneralVLMSpecialist
from geochat.constants import IMAGE_TOKEN_INDEX
from geochat.model.geochat_arch import GeoChatMetaForCausalLM


# =====================================================================
# Dummy components for isolated Test 3 (splicing)
# =====================================================================
class DummyConfig:
    def __init__(self, tune_mm_mlp_adapter=False, mm_use_im_start_end=False, pad_token_id=0):
        self.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        self.mm_use_im_start_end = mm_use_im_start_end
        self.pad_token_id = pad_token_id


class DummyInnerModel(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(35000, embed_dim)
        self.vision_tower = nn.Identity()
        self.mm_projector = nn.Linear(embed_dim, embed_dim)

    def get_vision_tower(self):
        return self.vision_tower


class DummyGeoChatModel(GeoChatMetaForCausalLM, nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.model = DummyInnerModel(embed_dim)
        self.config = DummyConfig()
        self.device = torch.device("cpu")

    def get_model(self):
        return self.model

    def encode_images(self, images):
        return torch.randn(images.shape[0], 10, 64)


# Shared specialist cache for performance across tests
_shared_specialist = None

def get_specialist():
    global _shared_specialist
    if _shared_specialist is None or not _shared_specialist._loaded:
        _shared_specialist = _load_geochat()
    return _shared_specialist


# =====================================================================
# FIX 1 REGRESSION TEST: _load_geochat returns GeoChatSpecialist
# =====================================================================
def test_fix1_loader_routing():
    print("\n[Regression Test 1] Verifying _load_geochat() loader routing...")
    manager = build_registry(device="cuda")
    assert manager._specs[GEOCHAT].loader_fn is _load_geochat, (
        f"FAILED: Expected _specs[GEOCHAT].loader_fn to be _load_geochat, got {manager._specs[GEOCHAT].loader_fn}"
    )

    specialist = get_specialist()
    assert isinstance(specialist, GeoChatSpecialist), (
        f"FAILED: Expected GeoChatSpecialist instance, got {type(specialist)}"
    )
    assert not isinstance(specialist, GeneralVLMSpecialist), (
        "FAILED: _load_geochat() returned GeneralVLMSpecialist!"
    )
    print("  PASSED: _load_geochat() returned valid GeoChatSpecialist.")


# =====================================================================
# FIX 2 REGRESSION TEST: GEOCHAT ModelSpec has shared_group=None
# =====================================================================
def test_fix2_shared_group_none():
    print("\n[Regression Test 2] Verifying GEOCHAT shared_group...")
    manager = build_registry(device="cuda")
    spec = manager._specs[GEOCHAT]

    assert spec.shared_group is None, (
        f"FAILED: GEOCHAT shared_group must be None, got '{spec.shared_group}'"
    )
    assert spec.shared_group != "qwen_vl", (
        "FAILED: GEOCHAT is still linked to 'qwen_vl' shared group!"
    )
    print(f"  PASSED: GEOCHAT shared_group is {spec.shared_group} (isolated lifecycle).")


# =====================================================================
# FIX 3 REGRESSION TEST: Splicing at position 0 has no wraparound
# =====================================================================
def test_fix3_splicing_position_0():
    print("\n[Regression Test 3] Verifying multimodal splicing at position 0...")
    model = DummyGeoChatModel(embed_dim=64)
    # [IMAGE_TOKEN_INDEX, 101, 102, 103, 104] -> 1 image token + 4 text tokens
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, 101, 102, 103, 104]], dtype=torch.long)
    images = torch.randn(1, 3, 224, 224)
    attention_mask = torch.ones_like(input_ids)

    out_ids, out_mask, out_pkv, out_embeds, out_labels = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=images,
    )

    # 10 synthetic image features + 4 text tokens = 14
    expected_length = 10 + 4
    actual_length = out_embeds.shape[1]

    assert actual_length == expected_length, (
        f"FAILED: Expected embedding length {expected_length}, got {actual_length} (wraparound bug)!"
    )
    print(f"  PASSED: Embeddings length {actual_length} == expected {expected_length} (no wraparound).")


# =====================================================================
# FIX 4 REGRESSION TEST: vision_tower has all 391 checkpoint keys
# =====================================================================
def test_fix4_vision_tower_keys():
    print("\n[Regression Test 4] Verifying vision tower checkpoint weights...")
    specialist = get_specialist()

    vt = specialist.model.get_vision_tower()
    named_params = list(vt.named_parameters())
    param_count = len(named_params)
    print(f"  Loaded vision tower named_parameters count: {param_count}")

    # Filter out bitsandbytes 4-bit quantization metadata if in 4-bit mode
    base_keys = [
        k for k in vt.state_dict().keys()
        if not any(k.endswith(x) for x in ['.absmax', '.quant_map', '.nested_absmax', '.nested_quant_map', '.bitsandbytes__nf4'])
    ]
    base_key_count = len(base_keys)
    print(f"  Loaded vision tower base state_dict key count: {base_key_count}")

    assert param_count == 391, (
        f"FAILED: Expected exactly 391 vision tower named parameters, got {param_count}!"
    )
    assert base_key_count == 391, (
        f"FAILED: Expected exactly 391 vision tower base keys, got {base_key_count}!"
    )
    print("  PASSED: Vision tower key count is exactly 391.")


# =====================================================================
# FIX 5 REGRESSION TEST: Features spliced > 0 and no "nobody is perfect"
# =====================================================================
def test_fix5_generation_splicing_and_output():
    print("\n[Regression Test 5] Verifying generation multimodal splicing and output...")
    test_image = os.path.join(PROJECT_ROOT, "data", "SECOND", "im2", "00003.png")
    assert os.path.exists(test_image), f"Test image missing: {test_image}"
    query = "What objects and features are visible in this satellite image?"

    specialist = get_specialist()

    # Hook prepare_inputs_labels_for_multimodal to capture feature splicing
    spliced_count = 0
    spliced_shapes = []
    orig_prepare = specialist.model.prepare_inputs_labels_for_multimodal

    def hook_prepare(input_ids, attention_mask, past_key_values, labels, images):
        nonlocal spliced_count
        ret = orig_prepare(input_ids, attention_mask, past_key_values, labels, images)
        new_ids, new_mask, new_pkv, new_embeds, new_labels = ret
        if new_embeds is not None:
            spliced_count += 1
            spliced_shapes.append(list(new_embeds.shape))
        return ret

    specialist.model.prepare_inputs_labels_for_multimodal = hook_prepare

    try:
        res = specialist.run(image=test_image, query=query, task_type="vqa")
    finally:
        specialist.model.prepare_inputs_labels_for_multimodal = orig_prepare

    answer = res["answer"]
    print(f"  Features spliced count: {spliced_count}")
    print(f"  Spliced shapes: {spliced_shapes}")
    print(f"  Decoded output: \"{answer}\"")

    assert spliced_count > 0, (
        f"FAILED: Features Spliced Count was {spliced_count} (image bypassed)!"
    )
    assert "nobody is perfect" not in answer.lower(), (
        f"FAILED: Output contains degenerate fallback text 'nobody is perfect'!"
    )
    assert len(answer.strip()) > 0, "FAILED: Output was empty!"

    # Check for zero CJK characters
    cjk_chars = [c for c in answer if '\u4e00' <= c <= '\u9fff']
    assert len(cjk_chars) == 0, (
        f"FAILED: Output contains {len(cjk_chars)} CJK characters: {cjk_chars}"
    )

    print("  PASSED: Multimodal features successfully spliced and output is coherent.")


def test_fix6_repetition_loop_prevention():
    """
    Test 6: Verify repetition loop prevention on high-density scenes (e.g. 00018.png)
    and verify the post-hoc degenerate repetition detector.
    """
    print("\n[Regression Test 6] Verifying repetition loop mitigation and detector...")
    specialist = get_specialist()

    # 1. Test post-hoc detector on synthetic loop
    mock_degenerate = (
        "In this satellite image, there are several objects visible, including a football field, "
        "a building, a building, a building, a building, a building, a building."
    )
    cleaned, is_deg = specialist._detect_and_clean_degenerate_repetition(mock_degenerate)
    assert is_deg is True, "FAILED: Post-hoc detector failed to catch consecutive repetition loop!"
    assert cleaned.count("a building") == 1, (
        f"FAILED: Expected exactly 1 mention of 'a building' after truncation, got: {cleaned}"
    )

    # 2. Test live generation on 00018.png
    test_image = os.path.join(PROJECT_ROOT, "data", "SECOND", "im2", "00018.png")
    query = "What objects and features are visible in this satellite image?"
    res = specialist.run(image=test_image, query=query, task_type="vqa")
    answer = res["answer"]
    tokens = specialist.tokenizer.encode(answer, add_special_tokens=False)

    print(f"  00018.png Decoded output: \"{answer}\"")
    print(f"  Token length: {len(tokens)}")
    print(f"  'a building' count: {answer.count('a building')}")

    assert answer.count("a building") < 3, (
        f"FAILED: 'a building' appeared {answer.count('a building')} times in output!"
    )
    assert len(tokens) < 150, (
        f"FAILED: Output ran on for {len(tokens)} tokens without terminating!"
    )
    assert len(answer.strip()) > 0, "FAILED: Output was empty!"
    print("  PASSED: Repetition loop prevented; 00018.png generated concise, non-repeating description.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING ALL 6 GEOCHAT REGRESSION TESTS STANDALONE")
    print("=" * 70)

    try:
        test_fix1_loader_routing()
        test_fix2_shared_group_none()
        test_fix3_splicing_position_0()
        test_fix4_vision_tower_keys()
        test_fix5_generation_splicing_and_output()
        test_fix6_repetition_loop_prevention()
        print("\n" + "=" * 70)
        print("ALL 6 GEOCHAT REGRESSION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
    finally:
        if _shared_specialist is not None and _shared_specialist._loaded:
            _shared_specialist.unload()
            gc.collect()
            torch.cuda.empty_cache()

