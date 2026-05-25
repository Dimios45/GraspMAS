import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer

VLM_MODEL_ID = "/mnt/data/mritunjoyh/models/Qwen2-VL-7B-Instruct"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        print(f"Loading {VLM_MODEL_ID} (first call — this takes a few minutes)...")
        _processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        _device = os.environ.get("VLM_DEVICE", "cuda:1")
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": _device}
        )
        _model.eval()
        print("VLM loaded.")


def _to_qwen_messages(messages):
    """Convert OpenAI-format messages to Qwen2-VL format."""
    out = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        else:
            parts = []
            for p in content:
                if p["type"] == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image_url":
                    parts.append({"type": "image", "image": p["image_url"]["url"]})
            out.append({"role": role, "content": parts})
    return out


def vlm_chat(messages, max_new_tokens=300, temperature=1.0, **kwargs):
    _load()
    msgs = _to_qwen_messages(messages)
    text = _processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = _processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(next(_model.parameters()).device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature > 0 and temperature != 1.0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    with torch.no_grad():
        ids = _model.generate(**inputs, **gen_kwargs)
    trimmed = ids[:, inputs.input_ids.shape[1]:]
    return _processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
