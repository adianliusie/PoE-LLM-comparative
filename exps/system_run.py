import os
import argparse
import pickle
import random
import openai
import os

from tqdm import tqdm
from datetime import datetime

from src.data_handler import DataHandler
from src.utils.general import save_json, load_json, load_pickle
from src.models import load_interface
from src.utils.post_processing import save_combined_json, delete_leftover_files
from src.utils.prompts import PROMPT_DICT

import time
# python system_run.py --output-path output_texts/falcon-7b/summeval-consistency/prompt_c1 --system falcon-7b --dataset summeval-s --score consistency --prompt-id c1 --shuffle --comparative
# python system_run.py --output-path output_texts/falcon-7b/summeval-consistency/prompt_c1 --system falcon-7b --dataset summeval-s --score-type consistency --prompt-id comparative-1 --shuffle --comparative
# python system_run.py --output-path output_texts2/chatgpt/summeval-relevance/comparative-1 --system chatgpt --dataset summeval-s --score-type relevance --prompt-id comparative-1 --shuffle --comparative

def main(
    system:str,
    output_path:str,
    dataset:str='summeval',
    score_type:str='consistency',
    shuffle:bool=True,
    comparative:bool=True,
    max_len:int=None,
    device:str=None,
    num_comparisons:int=None
):
    #load prompt from default, or choose your own prompt
    if comparative:
        prompt_id = f"{dataset}-{score_type[:3]}-comp"
        prompt_template = PROMPT_DICT[prompt_id]
    
    #< had a line here adding text to the end of decoder only models >

    # print path and template
    print('-'*50, '\n', output_path)
    print('-'*50, '\n', prompt_template, '\n', '-'*50)

    #< had a line here about caching inputs for podcast where there is a max_len >
    data_handler = DataHandler(prompt_template, dataset=dataset)
    if comparative:
        proc_inputs = data_handler.comparative_texts(score_type)
    else:
        proc_inputs = data_handler.scoring_texts(score_type)
    
    # create directory if not already existing
    system_output_path = f"{output_path}/outputs"
    if not os.path.isdir(system_output_path):
        os.makedirs(system_output_path)   

    if   dataset == 'summeval':    decoder_prefix='Summary'
    elif dataset == 'topicalchat': decoder_prefix='Response'

    # select the model to run the inputs through
    interface = load_interface(system=system, decoder_prefix=decoder_prefix, device=device)
    
    # save experiment settings 
    info = {
        'prompt_id':prompt_id,
        'prompt':prompt_template,
        'system':system, 
        'dataset':dataset,
        'score_type':score_type,
        'comparative':comparative,
        'decoder_prefix':decoder_prefix
    }             
            
    info_path = f"{output_path}/info.json"
    if not os.path.isfile(info_path):
        save_json(info, info_path) 
    else:
        log_info = load_json(info_path)
        assert all([info[k] == log_info[k] for k in info.keys() if k != 'dataset'])
        
    # select evaluation order for examples (shuffling allows parallelisation)
    ids = [x for x in range(len(proc_inputs))]
    if shuffle:
        random.shuffle(ids)

    # determine examples already run (and processed)
    done_ids = []
    if os.path.isfile(f"{system_output_path}/combined.json"):
        done_ids = set(list(load_json(f"{system_output_path}/combined.json").keys()))
    
    # process all inputs to chatgpt
    for k, idx in enumerate(ids):
        # break early if sufficient comparisons have been done
        if num_comparisons and (k + len(done_ids) > num_comparisons):
            break

        ex = proc_inputs[idx]
        outpath = f"{system_output_path}/{ex.ex_id}.json"

        # skip outputs already computed
        if (os.path.isfile(outpath)) or (ex.ex_id in done_ids):
            continue

        response = interface.prompt_classifier_response(
            input_text=ex.input_text, 
        )

        # get and print generated text
        gen_text = response.output_text
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"[{current_time}] {ex.ex_id} : {gen_text}")

        # save output file
        save_json(response.__dict__, outpath)

    save_combined_json(system_output_path)  
    delete_leftover_files(system_output_path)

def generation_parser():
    """ Build Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True, help='where to save chatgpt outputs')
    
    parser.add_argument('--system', type=str, default='flant5-large', help='which transformer to use')

    parser.add_argument('--dataset', type=str, default='summeval', help='which evaluation dataset to use')
    parser.add_argument('--score-type', type=str, default='consistency', help='which score to use of the dataset')
    
    parser.add_argument('--max-len', type=int, default=10, help='number of maximum tokens to be generated')
    parser.add_argument('--num-comparisons', type=int, default=None, help='number of comparisons to do for the dataset')

    parser.add_argument('--device', type=str, default=None, help='device to run experiments')

    parser.add_argument('--shuffle', action='store_true', help='whether to shuffling order of samples')
    parser.add_argument('--comparative', action='store_true', help='whether to do comparative evaluation')
    return parser

if __name__ == "__main__":
    parser = generation_parser()
    kwargs = vars(parser.parse_args())
    main(**kwargs)
