import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
from datasets import load_dataset


#processor = LlavaNextProcessor.from_pretrained("dillonlaird/hf-llava-v1.6-34b")

#model = LlavaNextForConditionalGeneration.from_pretrained("dillonlaird/hf-llava-v1.6-34b")#, device_map="auto")#, load_in_4bit=True)

model_id = "llava-hf/llava-v1.6-34b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    use_flash_attention_2=True
)



dset = load_dataset("jmhessel/newyorker_caption_contest", "ranking")

# Select a specific example from the test set
example = dset['train'][2]

# Print caption and other text data
print("Contest Number:", example['contest_number'])
print("Caption Choices:", example['caption_choices'])
print("Selected Caption (Label):", example['label'])
print("Image Description:", example['image_description'])
print("Uncanny Description:", example['image_uncanny_description'])

# Display the image
#display(example['image'])
print(type(example['contest_number']))


captions = example['caption_choices']
image = example['image'].convert("RGB")
#prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nProvide a detailed description of the cartoon image. Focus on identifying all visible elements, characters, actions, expressions, and setting. Do not interpret or explain the humor, simply describe what you see as if explaining the scene to someone who cannot see it.<|im_end|><|im_start|>assistant\n"

prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<|image|>\nProvide a detailed description of the cartoon image. Focus on identifying all visible elements, characters, actions, expressions, and setting. Do not interpret or explain the humor, simply describe what you see as if explaining the scene to someone who cannot see it.<|im_end|><|im_start|>assistant\n"


inputs = processor(prompt, image, return_tensors="pt", padding=True).to("cuda:0")  # Process inputs



# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

result = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]




print(len(result),result)
