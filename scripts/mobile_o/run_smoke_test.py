"""Mobile-O 0.5B smoke test (W2 gate, task #154).

Runs the three capability modes once each on GPU and records rough
latency (NOT benchmark-grade -- shared GPU, no warmup discipline).

Usage (from F:/work/Mobile-O, inside .venv-mobileo):
    .venv-mobileo/Scripts/python.exe run_smoke_test.py
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "checkpoints/Mobile-O-0.5B"
OUT_DIR = "smoke_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

results = {}


def step(name):
    def deco(fn):
        def wrapped(*a, **kw):
            t0 = time.time()
            try:
                out = fn(*a, **kw)
                dt = time.time() - t0
                results[name] = f"OK ({dt:.1f}s)"
                print(f"[{name}] OK in {dt:.1f}s", flush=True)
                return out
            except Exception as e:
                dt = time.time() - t0
                results[name] = f"FAIL ({type(e).__name__}: {str(e)[:120]})"
                print(f"[{name}] FAIL after {dt:.1f}s: {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()
                return None
        return wrapped
    return deco


@step("load")
def load_model():
    from mobileo.model.builder import load_pretrained_model
    tokenizer, model, _ = load_pretrained_model(MODEL_PATH)
    model.to("cuda:0")
    return tokenizer, model


@step("t2i_generation")
def t2i(tokenizer, model):
    from mobileo.constants import IMAGE_TOKEN_INDEX
    from mobileo.mm_utils import tokenizer_image_token
    from mobileo.conversation import conv_templates

    qs = ("Please generate image based on the following caption: "
          "a scarlet macaw perched on a moss-covered branch in a rainforest")
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda")
    img = model.generate_image(input_ids, pixel_values=None)[0]
    img.save(os.path.join(OUT_DIR, "smoke_t2i.png"))
    return img


@step("understanding")
def understanding(tokenizer, model):
    import torch
    from PIL import Image
    from mobileo.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN)
    from mobileo.mm_utils import tokenizer_image_token, process_images
    from mobileo.conversation import conv_templates

    image_processor = model.get_vision_tower().image_processor
    img_path = "assets/cute_cat.png"
    if not os.path.exists(img_path):
        # fall back to any asset png/jpg
        cands = [f for f in os.listdir("assets")
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not cands:
            raise FileNotFoundError("no test image in assets/")
        img_path = os.path.join("assets", cands[0])
    image = Image.open(img_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to("cuda", dtype=torch.float16)

    qs = DEFAULT_IMAGE_TOKEN + "\nWhat is in the image?"
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda")
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids, images=image_tensor, do_sample=False,
            max_new_tokens=64, use_cache=True)
    answer = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    print(f"  caption: {answer[:200]}", flush=True)
    with open(os.path.join(OUT_DIR, "smoke_understanding.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"image: {img_path}\nanswer: {answer}\n")
    return answer


@step("editing")
def editing(tokenizer, model):
    import torch
    from PIL import Image
    from mobileo.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN)
    from mobileo.mm_utils import tokenizer_image_token, process_images
    from mobileo.conversation import conv_templates

    image_processor = model.get_vision_tower().image_processor
    img_path = "assets/cute_cat.png"
    if not os.path.exists(img_path):
        cands = [f for f in os.listdir("assets")
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not cands:
            raise FileNotFoundError("no test image in assets/")
        img_path = os.path.join("assets", cands[0])
    image = Image.open(img_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to("cuda", dtype=torch.float16)

    qs = (DEFAULT_IMAGE_TOKEN +
          "\nPlease edit the image based on the following instruction: "
          "make it look like an oil painting")
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda")
    img = model.generate_image(input_ids, pixel_values=image_tensor)[0]
    img.save(os.path.join(OUT_DIR, "smoke_edit.png"))
    return img


def main():
    import torch
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    loaded = load_model()
    if loaded is None:
        print("\nLOAD FAILED -- aborting smoke test", flush=True)
    else:
        tokenizer, model = loaded
        print(f"  vision tower: {type(model.get_vision_tower()).__name__}",
              flush=True)
        t2i(tokenizer, model)
        understanding(tokenizer, model)
        editing(tokenizer, model)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\npeak VRAM: {peak:.1f} GB", flush=True)

    print("\n=== SMOKE TEST SUMMARY ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)
    n_fail = sum(1 for v in results.values() if v.startswith("FAIL"))
    print(f"SMOKE_{'FAIL' if n_fail else 'PASS'}", flush=True)


if __name__ == "__main__":
    main()
