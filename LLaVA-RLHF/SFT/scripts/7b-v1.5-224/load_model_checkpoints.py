from transformers import AutoModel, AutoTokenizer
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

#model_identifier = 'lmsys/vicuna-7b-v1.5'
#model_identifier = 'liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5' used for mm_checkpoint but no longer needed
model_identifier = 'lmsys/vicuna-7b-v1.3'
model = AutoModelForCausalLM.from_pretrained(model_identifier)
tokenizer = AutoTokenizer.from_pretrained(model_identifier)


#model.save_pretrained("/kuacc/users/dkukul19/hpc_run/LLaVA-RLHF/LLaVA-RLHF/LLaVA-RLHF_checkpoints_7b-1.3")

tokenizer.save_pretrained("/kuacc/users/dkukul19/hpc_run/LLaVA-RLHF/LLaVA-RLHF/LLaVA-RLHF_checkpoints_7b-1.3/vicuna-7b-v1.3")