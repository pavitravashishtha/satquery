import sys
sys.path.insert(0, '/home/pavitra/satquery/GeoChat')
sys.path.insert(0, '/home/pavitra/satquery')
import torch
import transformers
from geochat.model.language_model.geochat_llama import GeoChatLlamaForCausalLM, GeoChatConfig

config = GeoChatConfig(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=2,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)

model = GeoChatLlamaForCausalLM(config).to(device='cpu')

import functools

orig_prepare_multimodal = model.prepare_inputs_labels_for_multimodal
@functools.wraps(orig_prepare_multimodal)
def debug_multimodal(*args, **kwargs):
    print("DEBUG multimodal called with args:", [type(a) for a in args], "kwargs:", kwargs.keys())
    if len(args) > 0 and args[0] is not None:
        print("  input_ids shape:", args[0].shape)
    if 'input_ids' in kwargs and kwargs['input_ids'] is not None:
        print("  input_ids kwarg shape:", kwargs['input_ids'].shape)
    return orig_prepare_multimodal(*args, **kwargs)

model.prepare_inputs_labels_for_multimodal = debug_multimodal

orig_prepare_gen = model.prepare_inputs_for_generation
@functools.wraps(orig_prepare_gen)
def debug_gen(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    has_past_kv = False
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length"):
            has_past_kv = past_key_values.get_seq_length() > 0
        elif isinstance(past_key_values, (tuple, list)):
            has_past_kv = len(past_key_values) > 0

    if has_past_kv:
        input_ids = input_ids[:, -1:]

    if inputs_embeds is not None and not has_past_kv:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images", None),
        }
    )
    print("\nDEBUG prepare_inputs_for_generation called:")
    print("  has_past_kv:", has_past_kv)
    print("  returned input_ids shape:", getattr(model_inputs.get('input_ids'), 'shape', None))
    return model_inputs

model.prepare_inputs_for_generation = debug_gen

orig_forward = model.forward
call_cnt = 0
@functools.wraps(orig_forward)
def debug_fwd(input_ids=None, attention_mask=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, images=None, return_dict=None):
    global call_cnt
    call_cnt += 1
    print(f"\n--- FORWARD CALL #{call_cnt} ---")
    print("  input_ids:", getattr(input_ids, 'shape', None))
    print("  images:", getattr(images, 'shape', None))
    print("  past_key_values:", type(past_key_values), "len:", len(past_key_values) if past_key_values is not None else None)
    return orig_forward(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, images=images, return_dict=return_dict)

model.forward = debug_fwd

input_ids = torch.tensor([[1, 10, 20, 30, 40]], dtype=torch.long)
images = torch.randn(1, 3, 224, 224)

print("CALLING GENERATE:")
with torch.no_grad():
    model.generate(input_ids=input_ids, images=images, max_new_tokens=3, use_cache=True)
