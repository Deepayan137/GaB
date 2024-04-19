import pandas as pd
from pathlib import Path
import os
import fire

from llama import Llama
from typing import List
import csv
from tqdm import tqdm
import numpy as np
from copy import deepcopy


class LlamaQAGen(object):
	def __init__(self):

def  query_llama(queries: list[str],
    generator: Llama,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = None,
    max_batch_size: int = 4,
):

    
    # query llama
    
    dialogs = [[
	    {
	    	'role': 'system', 
	    	'content':"Strictly stick to the description and don't add details. Answer in the form DESC: caption"
	    	}, 
    	{
    		'role': 'user', 
    		'content': query}
    	] for query in queries]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    descriptions = []
    for idx_result, result in enumerate(results):
        answer = result['generation']['content'].strip()
        descriptions.append(answer.replace('\n', ' '))
   