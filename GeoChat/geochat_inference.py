"""
Direct GeoChat inference script - bypasses Gradio's broken template rendering entirely.
Uses GeoChat's own Chat class and loading pipeline directly, exactly as geochat_demo.py does internally.
"""

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from geochat.conversation import conv_templates, Chat
from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import get_model_name_from_path

# ---- Reproducibility (matches geochat_demo.py) ----
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = False
cudnn.deterministic = True

# ---- Configuration ----
MODEL_PATH = "MBZUAI/geochat-7B"
MODEL_BASE = None
DEVICE = "cuda"
GPU_ID = 0
LOAD_8BIT = False
LOAD_4BIT = True  # set True if you hit VRAM issues

# ---- Load the model (identical to what geochat_demo.py does) ----
print("Loading GeoChat model... (this can take a minute)")
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, MODEL_BASE, model_name, LOAD_8BIT, LOAD_4BIT, device=DEVICE
)
model = model.eval()
device = f"cuda:{GPU_ID}"

print(f"Memory used after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ---- Set up the Chat interface (this is GeoChat's own conversational wrapper) ----
CONV_VISION = conv_templates['llava_v1'].copy()
chat = Chat(model, image_processor, tokenizer, device=device)


def ask_geochat(image_path: str, question: str, temperature: float = 0.6, max_new_tokens: int = 300) -> str:
    """
    Send one image + one question to GeoChat and get back a plain-text answer.
    This mirrors exactly what geochat_demo.py's gradio_ask + gradio_stream_answer do,
    minus all the Gradio UI/visualization wrapping.
    """
    image = Image.open(image_path).convert("RGB")

    # Fresh conversation state for each independent question
    chat_state = CONV_VISION.copy()
    img_list = []

    # Upload image into the chat (this runs the vision encoder internally)
    chat.upload_img(image, chat_state, img_list)

    # Ask the question
    chat.ask(question, chat_state)

    # Encode image tensor if not already encoded
    if len(img_list) > 0 and not isinstance(img_list[0], torch.Tensor):
        chat.encode_img(img_list)

    # Stream the answer and collect it into one string
    streamer = chat.stream_answer(
        conv=chat_state,
        img_list=img_list,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_length=2000,
    )

    output = ""
    for new_text in streamer:
        output += new_text

    return output.strip()


if __name__ == "__main__":
    # ---- Quick test using one of GeoChat's own bundled demo images ----
    test_image = "demo_images/7292.JPG"
    test_question = "how many vehicles are there?"
    answer = ask_geochat(test_image, test_question)
    print(answer)

    print(f"\nImage: {test_image}")
    print(f"Question: {test_question}\n")

    answer = ask_geochat(test_image, test_question)

    print("ANSWER:")
    print(answer)
