from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig,AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoProcessor
import torch
from PIL import Image
import requests
from datasets import load_dataset
import json
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf", quantization_config=quantization_config, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attention_2=True)
#model.to("cuda:0") # don't use this line
dset = load_dataset("jmhessel/newyorker_caption_contest", "ranking")

## Select a specific example from the test set
#example = dset['train'][2]
#
## Print caption and other text data
#print("Contest Number:", example['contest_number'])
#print("Caption Choices:", example['caption_choices'])
#print("Selected Caption (Label):", example['label'])
#print("Image Description:", example['image_description'])
#print("Uncanny Description:", example['image_uncanny_description'])
#
## Display the image
##display(example['image'])
#print(type(example['contest_number']))
#
#
#captions = example['caption_choices']
results = []
processed_contest_numbers = set()
example_counter = 0
unique_data_count = 0

for example in dset['train']:
    # to process only 4 images
    if unique_data_count > 3:
      break
    example_counter+=1
    if example_counter%100==0:
      print(f"{example_counter} data points processed")
    if example['contest_number'] in processed_contest_numbers:
      continue
    unique_data_count+=1
    if unique_data_count%10==0:
      print(f"{unique_data_count} unique images processed")

    image = example['image'].convert("RGB")
    #prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nProvide a detailed description of the cartoon image. Focus on identifying all visible elements, characters, actions, expressions, and setting. Do not interpret or explain the humor, simply describe what you see as if explaining the scene to someone who cannot see it.<|im_end|><|im_start|>assistant\n"
    #prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nProvide a description of the cartoon image. Focus on identifying visible elements, characters, actions, expressions, and setting that seem most important. Do not interpret or explain the humor, simply describe what you see as if explaining the scene to someone who cannot see it. Do not use more than 3 sentences.<|im_end|><|im_start|>assistant\n"
    #prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nProvide a brief description of the cartoon image. Do not use more than 3 sentences.<|im_end|><|im_start|>assistant\n"
    prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nProvide a brief description of the cartoon image. Do not use more than 3 sentences. Do not interpret or explain the humor, simply describe what you see as if explaining the scene to someone who cannot see it.<|im_end|><|im_start|>assistant\n"

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")  # Process inputs
    outputs = model.generate(**inputs, max_new_tokens=200,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.9,
                            top_k=50,
                            top_p=0.95)

    explanations = [processor.decode(o, skip_special_tokens=True) for o in outputs]
    # Record the results for this contest_number
    result_entry = {
        "contest_number": example['contest_number']
    }
    for i, explanation in enumerate(explanations, start=1):
        result_entry[f"exp{i}"] = explanation

    results.append(result_entry)

    # Add the contest_number to the set of processed contest numbers
    processed_contest_numbers.add(example['contest_number'])


# Write results to a JSON file
with open('caption_contest_results4.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Completed generating results for all test set examples.")