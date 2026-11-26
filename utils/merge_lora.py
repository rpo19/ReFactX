import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_path = "/data/horse/ws/ripo631h-quokka/sft_output"
merged_path = "/data/horse/ws/ripo631h-quokka/sft_merged"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(base, adapter_path)

print("Merging...")
model = model.merge_and_unload()

print(f"Saving merged model to {merged_path}...")
model.save_pretrained(merged_path)

print("Saving tokenizer...")
tok = AutoTokenizer.from_pretrained(adapter_path)
tok.save_pretrained(merged_path)

print("Done!")
