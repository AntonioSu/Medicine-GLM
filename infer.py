from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
import torch


def load_lora_config(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"]
    )
    return get_peft_model(model, config)


checkpoint = "chatglm-6b"
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).half().cuda()
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

model = load_lora_config(model)
model.load_state_dict(torch.load(f"output/medicine-glm-lora.pt"), strict=False)

model.eval()
while True:
    text = input("request:")
    response, history = model.chat(tokenizer, text, history=[])
    print(response)
