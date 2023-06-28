import os
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import torch
from datasets import load_dataset, DatasetDict, Dataset
from ExtendTrainer import Trainer
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_src_length = 200
max_dst_length = 500
PROMPT_PATTERN = "问：{}"
SEP_PATTERN = "\n答： "

checkpoint = "thuglm"

model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
bos = tokenizer.bos_token_id
eop = tokenizer.eop_token_id
pad = tokenizer.pad_token_id
mask = tokenizer.mask_token_id
gmask = tokenizer.sp_tokenizer[tokenizer.gmask_token]


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


model = load_lora_config(model)


def create_prompt(question):
    return PROMPT_PATTERN.format(question), SEP_PATTERN


def create_prompt_ids(tokenizer, question, max_src_length):
    prompt, sep = create_prompt(question)
    sep_ids = tokenizer.encode(
        sep,
        add_special_tokens=True
    )
    sep_len = len(sep_ids)
    special_tokens_num = 2
    prompt_ids = tokenizer.encode(
        prompt,
        max_length=max_src_length - (sep_len - special_tokens_num),
        truncation=True,
        add_special_tokens=False
    )

    return prompt_ids + sep_ids


def create_inputs_and_labels(tokenizer, context, target, device):
    # print(f"question :{question}")
    # print(f"answer:{answer}")
    prompt = create_prompt_ids(tokenizer, context, max_src_length)
    completion = tokenizer.encode(
        target,
        max_length=max_dst_length,
        truncation=True,
        add_special_tokens=False
    )

    inputs = prompt + completion + [eop]
    labels = [-100] * len(prompt) + completion + [eop]

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return inputs, labels


def get_attention_mask(tokenizer, input_ids, device):
    seq = input_ids.tolist()
    context_len = seq.index(bos)
    seq_len = len(seq)
    attention_mask = torch.ones((seq_len, seq_len), device=device)
    attention_mask.tril_()
    attention_mask[..., :context_len] = 1
    attention_mask.unsqueeze_(0)
    attention_mask = (attention_mask < 0.5).bool()
    return attention_mask


def get_position_ids(tokenizer, input_ids, device, position_encoding_2d=True):
    seq = input_ids.tolist()
    context_len = seq.index(bos)
    seq_len = len(seq)

    mask_token = mask if mask in seq else gmask
    use_gmask = False if mask in seq else gmask

    mask_position = seq.index(mask_token)

    if position_encoding_2d:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        if not use_gmask:
            position_ids[context_len:] = mask_position
        block_position_ids = torch.cat((
            torch.zeros(context_len, dtype=torch.long, device=device),
            torch.arange(seq_len - context_len, dtype=torch.long, device=device) + 1
        ))
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        if not use_gmask:
            position_ids[context_len:] = mask_position

    return position_ids


test_file_list = "data/dev.json"
train_file_list = "data/train.json"
raw_datasets = load_dataset("json", data_files={
    'train': train_file_list, 'valid': test_file_list}, cache_dir="cache_data")
print(raw_datasets['train'][0])


class QADataset(Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item_data = self.data[index]
        tokenizer = self.tokenizer
        input_ids, labels = create_inputs_and_labels(
            tokenizer,
            device=device,
            **item_data
        )

        attention_mask = get_attention_mask(tokenizer, input_ids, device)
        position_ids = get_position_ids(tokenizer, input_ids, device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    position_ids = []

    for obj in batch:
        input_ids.append(obj['input_ids'])
        labels.append(obj['labels'])
        attention_mask.append(obj['attention_mask'])
        position_ids.append(obj['position_ids'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
        'position_ids': torch.stack(position_ids)
    }


model.to(device)

training_args = TrainingArguments(
    "output",
    fp16=True,
    save_steps=500,
    save_total_limit=3,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=1,
    learning_rate=1e-4,
    max_steps=1500,
    logging_steps=50,
    remove_unused_columns=False,
    seed=0,
    data_seed=0,
    group_by_length=False,
    dataloader_pin_memory=False
)

train_dataset = QADataset(raw_datasets['train'], tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=collate_fn,
    tokenizer=tokenizer
)

trainer.train()
