from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig, GenerationConfig, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling, DataCollatorWithPadding
from peft import PeftModel, PeftConfig, LoraConfig, AutoPeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training
import evaluate
import nltk
from trl import SFTTrainer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from auto_gptq import exllama_set_max_input_length
import os
import gradio as gr
import torch
# torch.cuda.empty_cache()


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

#Checkpoints

coder_checkpoint = "./01Coder-7Bv0.1"


"""#MODEL LOADING"""

tokenizer = AutoTokenizer.from_pretrained(coder_checkpoint)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
#quantization_config=bnb_config

model = AutoModelForCausalLM.from_pretrained(coder_checkpoint, device_map='auto')
model.config.pad_token_id = tokenizer.pad_token_id


#INFERENCE

def inference(command):
    
    prompt_temp = "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions."

    prompt = f"[INST]{prompt_temp} {command}[/INST] Code:"
    model_inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    split_text_p = text.split("Code:")
    return split_text_p[1].strip() if len(split_text_p) > 1 else "No output"

# Create Gradio Interface
iface = gr.Interface(
    fn=inference,
    inputs=[
        gr.Textbox(placeholder="Enter your code command here", label="Code Instruction", lines=2),
    ],
    outputs=gr.Textbox(placeholder="Generated code will appear here"),
    title="KNOW D CODE <0/1>",
    description= f'''Welcome to the 01-Coder-7B Playground! 
    
    💡This is the perfect place for you to write and test your code. Here are some tips to help you get started:

    1.Please specify the programming language you would like to use for your task. (Eg: "Write a program in Python" or "Create a Java class for...") \n
    2.If there are any external libraries or packages that are necessary for your task, please be sure to include them in your instructions. (Eg:"Use the pandas library to manipulate data" or "Create a web application using Gradio.") \n
    3.Try to be as clear and comprehensive as possible in your instructions. The more specific you are, the easier it will be for 01coder to generate the code you need. (Eg: Write a Python function that takes in a list of integers as input and returns the second largest number in the list.) \n

    We hope you enjoy using the 01-Coder-7B Playground!
    ''',
    theme="huggingface",
    examples=[
        ["Write a Python code to solve quadratic equations"],
        ["Write a Java program to create a class CAR with the attributes"],
        ["Write a CSS program to create a submit button for a webpage"],
        ["Write a SQL query to find the Second highest salary"],
    ],
)
# Launch the Gradio Interface
iface.launch(share=True)