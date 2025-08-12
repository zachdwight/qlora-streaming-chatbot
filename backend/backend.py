# backend/backend.py
import threading
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
# TextIteratorStreamer import fallback (different HF versions expose it in different places)
try:
    from transformers import TextIteratorStreamer
except Exception:
    from transformers.generation.streamers import TextIteratorStreamer

from peft import PeftModel
from starlette.responses import StreamingResponse
import uvicorn
import os

# === Config (edit paths as needed) ===
BASE_MODEL = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "./tinyllama-qlora-finetuned")
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.95

# === Load tokenizer and model ===
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

print("Creating bitsandbytes config...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # switch to torch.float16 if your GPU doesn't support bfloat16
)

print("Loading base model (this may take a while)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Determine device for inputs (model uses device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device for inputs:", device)

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


def build_prompt(question: str) -> str:
    return f"""### System:
You are a helpful assistant.
### User:
{question}
### Assistant:"""


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Simple non-streaming fallback endpoint
    prompt = build_prompt(req.message)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"reply": response}


@app.post("/api/stream")
async def stream_chat(req: ChatRequest):
    """
    Streams generated tokens back to the client as plain text chunks.
    The client should read response.body as a stream (ReadableStream).
    """
    prompt = build_prompt(req.message)
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to an available device (model is sharded across GPUs if device_map=auto)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # TextIteratorStreamer yields incremental text
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate_in_thread():
        # model.generate will call streamer as tokens are produced
        model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    thread = threading.Thread(target=generate_in_thread)
    thread.start()

    # Async generator that yields strings as they come
    def stream_generator():
        try:
            for chunk in streamer:
                # chunk is a string (piece of text). Yield it directly.
                yield chunk
        except GeneratorExit:
            # client disconnected
            pass
        finally:
            thread.join(timeout=1)

    # Return plain text stream (easier for fetch + ReadableStream on the frontend)
    return StreamingResponse(stream_generator(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")

