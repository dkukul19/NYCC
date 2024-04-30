from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig,AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoProcessor
import torch
from PIL import Image
import requests
from datasets import load_dataset
import json
from torch.nn import DataParallel
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf", quantization_config=quantization_config, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attention_2=True)
#model.to("cuda:0") # don't use this line
#model = DataParallel(model)
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
processed_contest_numbers.update([566, 186, 571, 704, 542, 230, 184, 570, 166, 599, 683, 526, 355, 539, 587, 447, 572, 618, 219, 519, 530, 556, 554, 449, 339, 58, 739, 213, 593, 61, 668, 350, 631, 515, 501, 718, 581, 82, 203, 89, 741, 425, 217, 705, 651, 275, 565, 392, 173, 623, 728, 734, 680, 654, 549, 183, 658, 15, 546, 454, 708, 547, 413, 226, 372, 237, 733, 78, 597, 580, 417, 263, 249, 86, 577, 463, 619, 589, 333, 3, 395, 146, 353, 740, 712, 532, 377, 7, 535, 702, 348, 457, 527, 27, 326, 115, 647, 330, 569, 182, 59, 250, 461, 584, 540, 736, 223, 737, 108, 252, 327, 151, 148, 191, 205, 142, 601, 155, 168, 145, 494, 394, 760, 541, 727, 450, 335, 320, 723, 2, 308, 692, 109, 755, 465, 585, 32, 511, 594, 245, 106, 400, 241, 352, 325, 83, 563, 667, 113, 274, 596, 582, 443, 319, 138, 172, 707, 753, 120, 529, 331, 675, 225, 749, 516, 51, 179, 384, 124, 638, 508, 738, 512, 72, 720, 552, 255, 312, 500, 763, 645, 185, 421, 364, 16, 711, 381, 192, 509, 686, 104, 41, 383, 715, 673, 732, 606, 211, 220, 528]
)

processed_contest_numbers.update([159, 375, 22, 189, 592, 522, 128, 31, 745, 247, 608, 344, 112, 595, 437, 125, 65, 246, 742, 627, 762, 495, 171, 324, 703, 71, 531, 152, 198, 401, 693, 689, 624, 96, 743, 36, 412, 13, 170, 507, 670, 137, 141, 722, 49, 560, 440, 157, 648, 674, 343, 687, 193, 559, 209, 181, 92, 407, 221, 253, 642, 336, 639, 422, 143, 218, 426, 62, 432, 272, 160, 337, 321, 79, 555, 25, 224, 591, 725, 322, 214, 688, 127, 445, 347, 354, 735, 227, 612, 380, 759, 318, 679, 503, 232, 626, 714, 158, 373, 719]
)
example_counter = 0
unique_data_count = 0

for example in dset['train']:
    # to process only 201 images
    #if unique_data_count > 99:
    #  break

    example_counter+=1
    if example_counter%100==0:
      print(f"{example_counter} data points processed",flush=True)
    if example['contest_number'] in processed_contest_numbers:
      continue
    unique_data_count+=1
    if unique_data_count%10==0:
      print(f"{unique_data_count} unique images processed",flush=True)

    image = example['image'].convert("RGB")
    #prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nProvide a detailed description of the cartoon image. Focus on identifying all visible elements, characters, actions, expressions, and setting. Do not interpret or explain the humor, simply describe what you see as if explaining the scene to someone who cannot see it.<|im_end|><|im_start|>assistant\n"
    #prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nProvide a description of the cartoon image. Focus on identifying visible elements, characters, actions, expressions, and setting that seem most important. Do not interpret or explain the humor, simply describe what you see as if explaining the scene to someone who cannot see it. Do not use more than 3 sentences.<|im_end|><|im_start|>assistant\n"
    #prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nProvide a brief description of the cartoon image. Do not use more than 3 sentences.<|im_end|><|im_start|>assistant\n"
    #prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nProvide a brief description of the cartoon image. Do not use more than 3 sentences. Do not interpret or explain the humor, simply describe what you see as if explaining the scene to someone who cannot see it.<|im_end|><|im_start|>assistant\n"
    prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nCaption the black-and-white sketch cartoon shortly.<|im_end|><|im_start|>assistant\n"
    #too long again #prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nDescribe the scene depicted in the cartoon briefly and accurately, focusing solely on the observable elements such as characters, actions, and settings. Avoid explaining or interpreting the humor or underlying themes. Provide a straightforward caption in one to two sentences.<|im_end|><|im_start|>assistant\n"
    #prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nDescribe the cartoon shortly.<|im_end|><|im_start|>assistant\n"
    #prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nDescribe the black-and-white sketch cartoon shortly.<|im_end|><|im_start|>assistant\n"

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")  # Process inputs
    outputs = model.generate(**inputs, max_new_tokens=200,
                            num_return_sequences=5,
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
with open('caption_contest_results[300-end].json', 'w') as f:
    json.dump(results, f, indent=4)

print("Completed generating results for all test set examples.")