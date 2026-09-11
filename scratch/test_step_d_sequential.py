import sys
import torch
from PIL import Image

sys.path.insert(0, "/home/pavitra/satquery/GeoChat")
sys.path.insert(0, "/home/pavitra/satquery")

from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import get_model_name_from_path, process_images_demo, tokenizer_image_token, KeywordsStoppingCriteria
from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from geochat.conversation import conv_templates

print("=" * 70)
print("STEP D: SEQUENTIAL RUN ON A SINGLE SHARED MODEL INSTANCE")
print("=" * 70)

model_path = "MBZUAI/geochat-7B"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name, load_8bit=False, load_4bit=True, device="cuda"
)
model = model.eval()

# Hooks
telemetry = {"vt": [], "proj": [], "prep": [], "spliced": []}
vt = model.get_vision_tower()
inner_vm = getattr(vt.vision_tower, "vision_model", vt.vision_tower)

def vt_hook(module, inputs, output):
    feat = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    telemetry["vt"].append({
        "mean": feat.float().mean().item(),
        "std": feat.float().std().item(),
        "shape": list(feat.shape),
    })
inner_vm.register_forward_hook(vt_hook)

proj = model.get_model().mm_projector
def proj_hook(module, inputs, output):
    telemetry["proj"].append({
        "mean": output.float().mean().item(),
        "std": output.float().std().item(),
        "shape": list(output.shape),
    })
proj.register_forward_hook(proj_hook)

def run_image(image_path, label):
    print(f"\n--- Running {label}: {image_path} ---")
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = process_images_demo([pil_image], image_processor).to(device="cuda", dtype=torch.float16)

    # 1. Direct encode hook
    vt_before = len(telemetry["vt"])
    proj_before = len(telemetry["proj"])
    with torch.no_grad():
        enc = model.encode_images(image_tensor)
    vt_after = len(telemetry["vt"])
    proj_after = len(telemetry["proj"])

    print(f"  Direct encode VT stats:   mean={telemetry['vt'][-1]['mean']:.8f}, std={telemetry['vt'][-1]['std']:.8f}")
    print(f"  Direct encode Proj stats: mean={telemetry['proj'][-1]['mean']:.8f}, std={telemetry['proj'][-1]['std']:.8f}")

    # 2. Generate pass
    query = "What objects and features are visible in this satellite image?"
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n")
    conv.messages[-1][1] = conv.messages[-1][1] + " " + query
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.where(input_ids == IMAGE_TOKEN_INDEX, torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype), input_ids)
    stop_str = conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    vt_gen_start = len(telemetry["vt"])
    proj_gen_start = len(telemetry["proj"])

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            past_key_values=None,
            do_sample=False,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            max_new_tokens=25,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    vt_gen_calls = len(telemetry["vt"]) - vt_gen_start
    proj_gen_calls = len(telemetry["proj"]) - proj_gen_start

    generated_ids = outputs[0, input_ids.shape[1]:]
    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if raw_answer.endswith(stop_str):
        raw_answer = raw_answer[:-len(stop_str)].strip()

    print(f"  Generate VT calls:   {vt_gen_calls}")
    print(f"  Generate Proj calls: {proj_gen_calls}")
    print(f"  Generate Output:     \"{raw_answer}\"")

run_image("data/SECOND/im2/00003.png", "CALL 1: Image 1")
run_image("data/SECOND/im2/00011.png", "CALL 2: Image 2 (reused instance)")
