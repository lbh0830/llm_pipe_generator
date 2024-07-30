import json

import pandas as pd
import accelerate
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, LlamaForCausalLM, pipeline
from tqdm.auto import tqdm
from transformers import GenerationConfig
from transformers.pipelines.pt_utils import KeyDataset
import os
import torch
import datetime

def auto_load_dataset(dataset_path: str, dataset_split: str = 'train'):
    if dataset_path.endswith('.csv'):
        return load_dataset('csv', data_files=dataset_path)[dataset_split]
    return load_dataset(dataset_path)[dataset_split]

def auto_load_dataframe(dataset_path: str):
    if dataset_path.endswith('.csv'):
        return pd.read_csv(dataset_path)
    elif dataset_path.endswith('.jsonl'):
        return pd.read_json(dataset_path, orient='records', lines=True)
    else:
        raise ValueError("Dataset path must be a csv or jsonl file")

def get_pipeline(path, tokenizer, accelerator):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    
    terminators = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    print(f"{terminators=}")
    # tokenizer.pad_token = tokenizer.eos_token

    generator = TextGenerationPipeline(model = model, tokenizer = tokenizer, num_workers=accelerator.state.num_processes*4, pad_token_id=tokenizer.pad_token_id, eos_token_id=terminators)

    return generator

def prompt_insert(content, full_prompt_file_path):
    with open(full_prompt_file_path, 'r', encoding="utf-8") as file:
        full_prompt = file.read()
    return full_prompt.format(CONTENT=content)

def generate_instruct(raw_df, column_name:str, tokenizer, system_prompt:str="", full_prompt_file_path:str=None):
    
    raw_ls = raw_df[column_name].tolist()
    instruction_ls = []
    for raw in tqdm(raw_ls, desc="Generating instruction"):
        chat_format = []
        if full_prompt_file_path is not None:
            instruction = prompt_insert(raw, full_prompt_file_path)
        else:
            instruction = raw
        if system_prompt != "":
            chat_format.append({"role": "system", "content": system_prompt})
        chat_format.append({"role": "user", "content": instruction})
        instruction_ls.append(tokenizer.apply_chat_template(chat_format, add_generation_prompt=True, tokenize=False))
    
    raw_df['instruction'] = instruction_ls
    return raw_df, Dataset.from_dict({"instruction": instruction_ls})

def append_to_dataset(output_path, instance):
    if os.path.isfile(output_path):
        with open(output_path, 'a', encoding="utf-8") as file:
            file.write(json.dumps(instance, ensure_ascii=False))
            file.write('\n')
    else:
        with open(output_path, 'w', encoding="utf-8") as file:
            file.write(json.dumps(instance, ensure_ascii=False))
            file.write('\n')

def get_data_token_length_filter(data, tokenizer, max_length=3072):
    filter_ls = []
    for i in tqdm(range(len(data)), desc="Filtering data"):
        if len(tokenizer(data[i]['instruction'])['input_ids']) > max_length:
            filter_ls.append(False)
        else:
            filter_ls.append(True)
    return filter_ls

@torch.inference_mode()
def main(model_path: str,
         dataset_path: str = None,
         output_path: str = None,
         full_prompt_file_path: str = None,
         column_name: str = None,
         system_prompt: str = "",
         generation_config: str = None,
         batch_size: int = 1,
         num_beams: int = 1,
         start_id: int = None,
         end_id: int = None,
         **kwargs):
    
    accelerator = accelerate.Accelerator()

    if generation_config is None:
        generation_config = {
            'temperature': 0.4,
            'top_k': 40,
            'top_p': 0.9,
            'do_sample': False,
            'num_beams': num_beams,
            'repetition_penalty': 1,
            'max_new_tokens': 1024
        }
        if accelerator.is_main_process:
            print('Doesn\'t detect generation config file, use default generation config:\n', generation_config)
    else:
        # 從generation_config.json讀取設定
        if accelerator.is_main_process:
            print(f'Loading generation config from:\n{generation_config}')
        with open(generation_config, "r") as f:
            generation_config = json.load(f)
        if accelerator.is_main_process:
            print(generation_config)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', use_fast=False)
    pipe = get_pipeline(model_path, tokenizer, accelerator)
    
    raw_df = auto_load_dataframe(dataset_path)
    raw_df, instruction_ds = generate_instruct(raw_df, column_name, tokenizer, system_prompt, full_prompt_file_path)

    if start_id is not None and end_id is not None:
        raw_df = raw_df.iloc[start_id:end_id]
        instruction_ds = instruction_ds.select(range(start_id, end_id))

    if accelerator.is_main_process:
        print(f"raw file:\n{raw_df.iloc[0]}")
        print(f"instruction:\n{instruction_ds[0]}")

    id = 0

    for res in tqdm(pipe(KeyDataset(instruction_ds, "instruction"), return_full_text=False, batch_size=batch_size, **generation_config), total=len(raw_df)):
        for r in res:
            # 取得raw_df特定id的所有內容
            instance = raw_df.iloc[id].to_dict()
            instance['resp'] = r['generated_text']
            id+=1
            append_to_dataset(output_path, instance)

if __name__ == '__main__':
    import fire
    fire.Fire(main)