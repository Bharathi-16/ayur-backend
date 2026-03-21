"""
Inference Engine — Model loading, device detection, streaming generation
"""
import json
import time
import threading
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# ── Model State (global singleton) ──
model_state = {
    "loaded": False,
    "loading": False,
    "message": "Not started",
    "progress": 0,
    "error": None,
    "device": None,
    "model_id": None,
}

tokenizer = None
model = None
_stop_event = threading.Event()

# ── Default Config ──
MODEL_ID = "bharatgenai/AyurParam"

DEFAULT_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.6,
    "top_p": 0.85,
    "top_k": 40,
    "repetition_penalty": 1.05,
    "do_sample": True,
}

PRESETS = {
    "fast": {
        "max_new_tokens": 80,
        "temperature": 0.3,
        "top_p": 0.9,
        "do_sample": False,
    },
    "quality": {
        "max_new_tokens": 300,
        "temperature": 0.65,
        "top_p": 0.85,
        "top_k": 40,
        "do_sample": True,
    },
    "balanced": DEFAULT_CONFIG.copy(),
}

SYSTEM_PROMPT = (
    "You are AyurParam, an expert AI assistant specialising in Ayurveda. "
    "Provide accurate, evidence-based guidance on herbs, doshas, and lifestyle. "
    "Always be compassionate and clear. Keep responses helpful. "
    "If asked about a disease, structure your answer with: Nidana (causes), "
    "Lakshana (symptoms), Chikitsa (treatment), Pathya (beneficial), Apathya (avoid)."
)


def get_status():
    return {
        "state": "ready" if model_state["loaded"] else ("error" if model_state["error"] else "loading"),
        "message": model_state["message"],
        "progress": model_state["progress"],
        "device": model_state["device"],
        "model_id": model_state["model_id"],
    }


def start_model_loading():
    """Start model loading in background thread"""
    if model_state["loading"] or model_state["loaded"]:
        return
    model_state["loading"] = True
    threading.Thread(target=_load_model_task, daemon=True).start()


def reload_model():
    """Hot-reload: unload and re-load"""
    global model, tokenizer
    model_state["loaded"] = False
    model_state["loading"] = False
    model_state["error"] = None
    model = None
    tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    start_model_loading()


def _load_model_task():
    global tokenizer, model
    try:
        import os
        import traceback
        from transformers import AutoConfig
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print("[Inference] Starting model load task...")
        
        model_state["message"] = f"Loading tokenizer for {MODEL_ID}..."
        model_state["progress"] = 20
        model_state["model_id"] = MODEL_ID

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("[Inference] Tokenizer loaded successfully.")
        model_state["message"] = "Detecting hardware..."
        model_state["progress"] = 40

        # Device detection
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif platform.machine() == "arm64" and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            # CPU fallback: Using float16 instead of float32 to save 50% RAM.
            # Note: Some older CPUs might need float32 for stability, but float16 is required for memory limits.
            device = "cpu"
            dtype = torch.float16


        model_state["device"] = device.upper()
        model_state["message"] = f"Loading model on {device.upper()} (this may take a minute)..."
        model_state["progress"] = 60
        print(f"[Inference] Chosen Device: {device.upper()}")

        # Optimization: Use all available CPU cores for torch operations
        if device == "cpu":
            torch.set_num_threads(os.cpu_count())

        # Load config and fix rope_scaling issues common with older custom models in new transformers versions
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        config.rope_scaling = None

        print(f"[Inference] Downloading/loading {MODEL_ID} weights...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device
        )
        model.eval()

        model_state["loaded"] = True
        model_state["loading"] = False
        model_state["message"] = f"Ready — {MODEL_ID} on {device.upper()}"
        model_state["progress"] = 100
        print(f"[Inference] ✅ Model loaded on {device.upper()}")

    except Exception as e:
        traceback.print_exc()
        model_state["error"] = str(e)
        model_state["loading"] = False
        model_state["message"] = f"Error: {str(e)}"
        print(f"[Inference] ❌ Failed: {e}")


def build_prompt(history, system_prompt=None, max_turns=5):
    """Build prompt string from conversation history"""
    sp = system_prompt or SYSTEM_PROMPT
    prompt = f"<system_prompt> {sp} "
    for h in history[-max_turns:]:
        role = "user" if h["role"] == "user" else "assistant"
        prompt += f"<{role}> {h['content']} "
    prompt += "<assistant>"
    return prompt


def stop_generation():
    """Signal to stop current generation"""
    _stop_event.set()


def generate_stream(history, config=None, system_prompt=None):
    """
    Generator that yields SSE-formatted token events.
    config: dict overriding DEFAULT_CONFIG keys
    """
    if not model_state["loaded"]:
        if not model_state["loading"]:
            from app.services.inference import start_model_loading
            start_model_loading()
        yield f"data: {json.dumps({'error': 'Model loading in progress... please try again in a moment.'})}\n\n"
        return

    _stop_event.clear()

    merged_config = DEFAULT_CONFIG.copy()
    if config:
        merged_config.update(config)

    prompt = build_prompt(history, system_prompt)
    start_time = time.time()

    try:
        with torch.inference_mode():
            device = model.device
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=int(merged_config.get("max_new_tokens", 150)),
                do_sample=bool(merged_config.get("do_sample", False)),
                temperature=float(merged_config.get("temperature", 0.6)),
                top_p=float(merged_config.get("top_p", 0.85)),
                top_k=int(merged_config.get("top_k", 40)),
                repetition_penalty=float(merged_config.get("repetition_penalty", 1.05)),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

            print(f"[Inference] Generating for prompt... (max_new_tokens={gen_kwargs['max_new_tokens']})")
            # Start generation thread
            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            full_text = ""
            token_count = 0

            for token in streamer:
                if _stop_event.is_set():
                    print("[Inference] Generation STOPPED by user.")
                    yield f"data: {json.dumps({'stopped': True, 'full': full_text.strip()})}\n\n"
                    return
                full_text += token
                token_count += 1
                if token_count % 10 == 0:
                    print(f"[Inference] Generated {token_count} tokens...")
                yield f"data: {json.dumps({'delta': token})}\n\n"

            elapsed = time.time() - start_time
            tokens_per_sec = round(token_count / elapsed, 1) if elapsed > 0 else 0
            print(f"[Inference] Generation DONE. Total tokens: {token_count}, Speed: {tokens_per_sec} t/s")

            yield f"data: {json.dumps({'done': True, 'full': full_text.strip(), 'token_count': token_count, 'latency_ms': int(elapsed * 1000), 'tokens_per_sec': tokens_per_sec})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
