# from unsloth import FastLanguageModel 
# from unsloth import is_bfloat16_supported
# import torch
# from trl import SFTTrainer
# from transformers import TrainingArguments
# from datasets import load_dataset
# max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/llama-3-70b-bnb-4bit",
#     max_length = max_seq_length

# FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# def generate_text(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(inputs["input_ids"], max_length=max_seq_length)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return generated_text

# prompt = "Explain the significance of machine learning in healthcare."
# generated_output = generate_text(prompt)
# print("Generated Text:", generated_output)

