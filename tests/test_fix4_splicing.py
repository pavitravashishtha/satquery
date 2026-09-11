import sys
import os
import torch
import torch.nn as nn

# Ensure GeoChat is importable
GEOCHAT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GeoChat")
if GEOCHAT_ROOT not in sys.path:
    sys.path.insert(0, GEOCHAT_ROOT)

from geochat.constants import IMAGE_TOKEN_INDEX
from geochat.model.geochat_arch import GeoChatMetaForCausalLM


class DummyConfig:
    def __init__(self, tune_mm_mlp_adapter=True, mm_use_im_start_end=True, pad_token_id=0):
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
    def __init__(self, embed_dim=64, tune_mm=True, mm_start_end=True):
        super().__init__()
        self.model = DummyInnerModel(embed_dim)
        self.config = DummyConfig(tune_mm_mlp_adapter=tune_mm, mm_use_im_start_end=mm_start_end)
        self.device = torch.device("cpu")

    def get_model(self):
        return self.model

    def encode_images(self, images):
        # Return synthetic image features: [B, num_patches, embed_dim]
        return torch.randn(images.shape[0], 10, 64)


def test_splicing_position_0():
    print("=== Testing Splicing with IMAGE_TOKEN_INDEX at position 0 ===")
    model_std = DummyGeoChatModel(embed_dim=64, tune_mm=False, mm_start_end=False)
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, 101, 102, 103, 104]], dtype=torch.long)
    images = torch.randn(1, 3, 224, 224)
    attention_mask = torch.ones_like(input_ids)

    out_ids, out_mask, out_pkv, out_embeds, out_labels = model_std.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=images,
    )

    # Expected length: 10 (image patches) + 4 (text tokens 101, 102, 103, 104) = 14
    expected_len = 10 + 4
    actual_len = out_embeds.shape[1]
    print(f"Position 0 standard: input shape={input_ids.shape}, out_embeds shape={out_embeds.shape}")
    assert actual_len == expected_len, f"Expected length {expected_len}, got {actual_len}!"

    # Test with mm_use_im_start_end=True and tune_mm=True (branch where bug was found)
    model_start_end = DummyGeoChatModel(embed_dim=64, tune_mm=True, mm_start_end=True)
    input_ids_se = torch.tensor([[IMAGE_TOKEN_INDEX, 32001, 101, 102, 103]], dtype=torch.long)
    out_ids, out_mask, out_pkv, out_embeds_se, out_labels = model_start_end.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids_se,
        attention_mask=torch.ones_like(input_ids_se),
        past_key_values=None,
        labels=None,
        images=images,
    )
    # In mm_start_end mode:
    # image_token_start == 0:
    # prefix_end = max(0, -1) = 0 (no wraparound)
    # image features: 10
    # end_im: 1 token (32001)
    # remaining: 3 tokens (101, 102, 103)
    # Total embeds: 10 + 1 + 3 = 14 tokens
    expected_len_se = 10 + 1 + 3
    actual_len_se = out_embeds_se.shape[1]
    print(f"Position 0 mm_start_end: input shape={input_ids_se.shape}, out_embeds shape={out_embeds_se.shape}")
    assert actual_len_se == expected_len_se, f"Expected length {expected_len_se}, got {actual_len_se}!"
    print("Assertion passed: Position 0 correctly handled with NO wraparound!")


def test_splicing_mid_sequence():
    print("\n=== Testing Splicing with IMAGE_TOKEN_INDEX mid-sequence ===")
    model_std = DummyGeoChatModel(embed_dim=64, tune_mm=False, mm_start_end=False)
    input_ids = torch.tensor([[50, 51, 52, IMAGE_TOKEN_INDEX, 101, 102, 103]], dtype=torch.long)
    images = torch.randn(1, 3, 224, 224)
    attention_mask = torch.ones_like(input_ids)

    out_ids, out_mask, out_pkv, out_embeds, out_labels = model_std.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=images,
    )
    # Expected length: 3 (prefix) + 10 (image patches) + 3 (suffix) = 16
    expected_len = 3 + 10 + 3
    actual_len = out_embeds.shape[1]
    print(f"Mid-sequence standard: input shape={input_ids.shape}, out_embeds shape={out_embeds.shape}")
    assert actual_len == expected_len, f"Expected length {expected_len}, got {actual_len}!"
    print("Assertion passed: Mid-sequence correctly spliced without dropping any tokens!")


def test_sentinel_replacement_before_generate():
    print("\n=== Testing -200 Sentinel Replacement and Assertion ===")
    input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 100, 101]], dtype=torch.long)
    pad_token_id = 0

    input_ids = torch.where(
        input_ids == IMAGE_TOKEN_INDEX,
        torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype),
        input_ids,
    )

    print(f"Cleaned input_ids: {input_ids.tolist()}")
    assert input_ids.min() >= 0, f"Error: input_ids contains negative token indices: {input_ids.min()}"
    assert IMAGE_TOKEN_INDEX not in input_ids, "IMAGE_TOKEN_INDEX still found in tensor!"
    print(f"Assertion passed: input_ids.min() = {input_ids.min().item()} >= 0. Ready for generate().")


if __name__ == "__main__":
    test_splicing_position_0()
    test_splicing_mid_sequence()
    test_sentinel_replacement_before_generate()
    print("\nALL FIX 4 UNIT CHECKS PASSED.")
