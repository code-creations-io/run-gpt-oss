from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "openai/gpt-oss-20b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load model directly onto GPU in bf16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": 0}  # Force full model onto CUDA:0
)

# Create inference pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device=0  # ensure generation runs on CUDA
)

# Chat prompt
chat = [
    {"role": "system", "content": "Reasoning: high"},
    {"role": "user", "content": "Explain the core concepts of string theory in 3 bullet points."},
]
formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Inference
outputs = pipe(formatted_prompt, max_new_tokens=128)
print(outputs[0]["generated_text"])
