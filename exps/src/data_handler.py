import json
import os
import numpy as np
import pandas as pd

from datasets import load_dataset
from types import SimpleNamespace
from typing import List
from functools import lru_cache
from tqdm import tqdm

from .utils.general import load_json_files

class DataHandler:
    def __init__(self, prompt_template:str, dataset:str='summeval', max_input_len:int=None):
        self.prompt_template = prompt_template
        self.documents = self.load_data(dataset)
        self.max_len = max_input_len

    def scoring_texts(self, score_type):
        outputs = []
        for doc in self.documents:
            num_responses = len(doc.responses)
            for k in range(num_responses):
                # relevant information need for text filling
                context = doc.context
                response = doc.responses[k]
                fact = getattr(doc, 'fact', None)

                # fill in the prompt template
                text_info = SimpleNamespace(
                    context=context,
                    response_A=response,
                    fact=fact
                )

                # get prompt input text
                input_text = self.fill_template(text_info) if self.prompt_template else None

                # get labels for scoring
                label = doc.scores[score_type][k]

                # add example to output
                ex_id = doc.context_id + '-' + str(k)
                ex = SimpleNamespace(
                    ex_id=ex_id,
                    input_text=input_text,
                    label=label,
                    response=response,
                    reference=getattr(doc, 'reference', None),
                )
                outputs.append(ex)
        return outputs

    def comparative_texts(self, score_type):
        outputs = []
        for doc in self.documents:
            num_responses = len(doc.responses)
            for i in range(num_responses):
                for j in range(num_responses):
                    # skip the same document
                    if i == j: continue

                    # relevant information need for text filling
                    context = doc.context
                    response_A = doc.responses[i]
                    response_B = doc.responses[j]
                    fact = getattr(doc, 'fact', None)

                    # fill in the prompt template
                    text_info = SimpleNamespace(
                        context=context,
                        response_A=response_A,
                        response_B=response_B,
                        fact=fact
                    )
                    input_text = self.fill_template(text_info)

                    # get comparative labels
                    score_1 = doc.scores[score_type][i]
                    score_2 = doc.scores[score_type][j]
                    score_diff = score_1-score_2

                    if   score_diff  > 0: label = 0
                    elif score_diff  < 0: label = 1
                    elif score_diff == 0: label = -1

                    # add example to output
                    ex_id = doc.context_id + '-' + str(i) + '-' + str(j)
                    ex = SimpleNamespace(
                        ex_id=ex_id,
                        input_text=input_text,
                        label=label,
                        score_diff=score_diff
                    )
                    outputs.append(ex)
        return outputs

    def comparative_texts_pair_s(self, score_type):
        # same but for some reason they want self-comparison too?
        outputs = []
        for doc in self.documents:
            num_responses = len(doc.responses)
            for i in range(num_responses):
                for j in range(num_responses):
                    # relevant information need for text filling
                    context = doc.context
                    response_A = doc.responses[i]
                    response_B = doc.responses[j]
                    fact = getattr(doc, 'fact', None)

                    # fill in the prompt template
                    text_info = SimpleNamespace(
                        context=context,
                        response_A=response_A,
                        response_B=response_B,
                        fact=fact
                    )
                    input_text = self.fill_template(text_info)

                    # get comparative labels
                    score_1 = doc.scores[score_type][i]
                    score_2 = doc.scores[score_type][j]
                    score_diff = score_1-score_2

                    if   score_diff  > 0: label = 0
                    elif score_diff  < 0: label = 1
                    elif score_diff == 0: label = -1

                    # add example to output
                    ex_id = doc.context_id + '-' + str(i) + '-' + str(j)
                    ex = SimpleNamespace(
                        ex_id=ex_id,
                        input_text=input_text,
                        label=label,
                        score_diff=score_diff
                    )
                    outputs.append(ex)
        return outputs

    def fill_template(self, text_info):
        text = self.prompt_template
        if '<A>' in text:
            text = text.replace('<A>', text_info.response_A)
        if '<B>' in text:
            text = text.replace('<B>', text_info.response_B)
        if '<topic>' in text:
            text = text.replace('<topic>', text_info.topic)
        if '<fact>' in text:
            text = text.prompt_template.replace("<fact>", text_info.fact)

        # truncate context if necessary
        if self.max_len: 
            num_ctx_tokens = self.max_len - len(self.tokenizer(text).input_ids)
            ctx_tokens = self.tokenizer(text_info.context).input_ids[:num_ctx_tokens]
            text_info.context = self.tokenizer.decode(ctx_tokens)
    
        if '<context>' in text:
            text = text.replace('<context>', text_info.context)

        return text

    #== Data Loading Methods ===========================================================#
    @classmethod
    def load_data(cls, dataset):
        if   dataset=='summeval':    documents = cls.load_summeval()
        elif dataset=='summeval-s':  documents = cls.load_summeval()[:20]
        elif dataset=='summeval-t':  documents = cls.load_summeval()[:5]
        elif dataset=='podcast':     documents = cls.load_podcast()
        elif dataset=='topicalchat': documents = cls.load_topicalchat()
        elif dataset=='webnlg':      documents = cls.load_webnlg()
        elif dataset=='wi-train':    documents = cls.load_write_and_improve(split='train')
        elif dataset=='wi-dev':      documents = cls.load_write_and_improve(split='dev')
        elif dataset=='hanna':       documents = cls.load_hanna()
        elif dataset=='cmcqrd':      documents = cls.load_cmcqrd()
        return documents

    @staticmethod
    @lru_cache(maxsize=3)
    def load_summeval()->List[SimpleNamespace]:
        output = []
        summ_eval = load_dataset('mteb/summeval')['test']
        for k, row in enumerate(summ_eval):
            ex = SimpleNamespace(
                context_id=str(k),
                context=row['text'],
                responses=row['machine_summaries'],
                reference=row['human_summaries'][0],
                scores={
                    'coherency':row['coherence'],
                    'fluency':row['fluency'],
                    'consistency':row['consistency'],
                    'relevance':row['relevance']
                }
            )
            output.append(ex)
        return output

    @staticmethod
    def load_topicalchat() -> List[SimpleNamespace]:
        data_path = "PATH_TO_PROCESSED_TOPICALCHAT_DATA"
        with open(data_path, "r") as f:
            x = f.read()
        data = json.loads(x)

        output = []
        for k, row in enumerate(data):
            responses = row['responses']
            ex = SimpleNamespace(
                context_id=str(k),
                context=row['context'],
                responses=[x['response'] for x in responses],
                fact=row['fact'],
                scores={
                    'coherency': [np.mean(x['Understandable']) for x in responses],
                    'naturalness': [np.mean(x['Natural']) for x in responses],
                    'continuity': [np.mean(x['Maintains Context']) for x in responses],
                    'engagingness': [np.mean(x['Engaging']) for x in responses],
                    'groundedness': [np.mean(x['Uses Knowledge']) for x in responses],
                    'overall': [np.mean(x['Overall']) for x in responses],
                }
            )
            output.append(ex)
        return output

    @staticmethod
    def load_webnlg() -> List[SimpleNamespace]:
        # dataset downloaded from https://github.com/ufal/nlgi_eval
        data_path = "PATH_TO_PROCESSED_WEBNLG_DATA"
        with open(data_path, "r") as f:
            x = f.read()
        data = json.loads(x)

        output = []
        for k, row in data.items():
            generated_texts, fluency, grammar, semantics = [], [], [], []
            for system, value in row.items():
                generated_texts, fluency, grammar, semantics = [], [], [], []
                bleu, meteor, ter = [], [], []
                for system, value in row.items():
                    generated_texts.append(value['text'])
                    fluency.append(value['fluency'])
                    grammar.append(value['grammar'])
                    semantics.append(value['semantics'])
                    bleu.append(value['bleu'])
                    meteor.append(value['meteor'])
                    ter.append(value['ter'])
                    triples = value['data'] # triples concatenated as string- same for all systems
                
                context = f"The following are semantic triples of the form (subject|relation|object)\n\n{triples}"
                ex = SimpleNamespace(
                    context_id=str(k),
                    context=context, 
                    responses=generated_texts,
                    scores={
                        'fluency': fluency,
                        'grammar': grammar,
                        'semantic': semantics,
                        'bleu': bleu,
                        'meteor': meteor,
                        'ter': ter
                    }
                )
            output.append(ex)
        return output

    @staticmethod
    def load_podcast()->List[SimpleNamespace]:
        podcast_data = load_dataset("potsawee/podcast_summary_assessment")['evaluation']
        system_ids = ['R1'] + [f"E{k}" for k in range(1,4)] + [f"A{k}" for k in range(1,17)]
        system2id = {v:k for k, v in enumerate(system_ids)}

        episodes = set(row['episode_id'] for row in podcast_data)
        episode2id = {v:str(k) for k, v in enumerate(episodes)}

        # splitting 3580 -> 179 * 20
        podcast_179 = {}
        score_mapping = {'B':0, 'F': 1, 'G': 2, 'E': 3} # Bad, Fair, Good, Excellent
        for row in podcast_data:
            episode_id = row['episode_id']
            system_id = row['system_id']
            if episode_id not in podcast_179:
                podcast_179[episode_id] = SimpleNamespace(
                    context_id=episode2id[row['episode_id']],
                    #context_id=row['episode_id'],
                    context=row['transcript'],
                    responses=[None for _ in range(20)],
                    scores={'overall': [None for _ in range(20)]},
                )
            # assert podcast_179[episode_id].context_id == row['episode_id'] # sanity check
            # assert podcast_179[episode_id].context == row['transcript'] # sanity check
            podcast_179[episode_id].responses[system2id[system_id]] = row['summary']
            podcast_179[episode_id].scores['overall'][system2id[system_id]] = score_mapping[row['score']]

        podcast_179 = [v for v in podcast_179.values()]
        return podcast_179
    
    @staticmethod
    def load_hanna()->List[SimpleNamespace]:
        file_path='PATH_TO_PROCESSED_HANNA_DATA'
        dataset = pd.read_csv(file_path)

        processed_df = {}
        for i in range(dataset.shape[0]):
            idx = dataset['Story ID'][i]
            if str(idx) not in processed_df:
                processed_df[str(idx)] = {
                    'input': dataset['Prompt'][i],
                    'output': 'Prompt: ' + dataset['Prompt'][i] + '\n\n' + 'Generated Story: ' + dataset['Story'][i],
                    'relevance': [dataset['Relevance'][i]],
                    'coherence': [dataset['Coherence'][i]],
                    'empathy': [dataset['Empathy'][i]],
                    'surprise': [dataset['Surprise'][i]],
                    'engagement': [dataset['Engagement'][i]],
                    'complexity': [dataset['Complexity'][i]]
                }
            else:
                processed_df[str(idx)]['relevance'].append(dataset['Relevance'][i])
                processed_df[str(idx)]['coherence'].append(dataset['Coherence'][i])
                processed_df[str(idx)]['empathy'].append(dataset['Empathy'][i])
                processed_df[str(idx)]['surprise'].append(dataset['Surprise'][i])
                processed_df[str(idx)]['engagement'].append(dataset['Engagement'][i])
                processed_df[str(idx)]['complexity'].append(dataset['Complexity'][i])

        inputs = [dp['input'] for dp in list(processed_df.values())]
        responses = [dp['output'] for dp in list(processed_df.values())]

        relevance = [round(np.mean(dp['relevance']),1) for dp in list(processed_df.values())]
        coherence = [round(np.mean(dp['coherence']),1) for dp in list(processed_df.values())]
        empathy = [round(np.mean(dp['empathy']),1) for dp in list(processed_df.values())]
        surprise = [round(np.mean(dp['surprise']),1) for dp in list(processed_df.values())]
        engagement = [round(np.mean(dp['engagement']),1) for dp in list(processed_df.values())]
        complexity = [round(np.mean(dp['complexity']),1) for dp in list(processed_df.values())]

        out = SimpleNamespace(
            context_id='0',
            context=None,
            responses=responses,
            scores={
                'coherence':coherence,
                'complexity':complexity,
                'empathy':empathy,
                'engagement':engagement,
                'relevance':relevance,
                'surprise':surprise
            }
        )

        return [out]

    @staticmethod
    def load_cmcqrd()->List[SimpleNamespace]:
        file_path='PATH_TO_PROCESSED_CMCQRD_DATA'

        with open(file_path, 'r') as f:
            data = json.load(f)

        questions = []
        difficulty_labels = []
        for ctx in data:
            context = ctx['context']
            for ex in ctx["questions"]:
                question=ex['question']
                options=ex['options']

                numerated_options = "\n".join([f"{i+1}) {x}" for i, x in enumerate(options)])

                full_question = context + '\n\n' + question + '\n' + numerated_options
                difficulty_label = ex['question_difficulty']

                questions.append(full_question)
                difficulty_labels.append(difficulty_label)

        out = SimpleNamespace(
            context_id='0',
            context=None,
            responses=questions,
            scores={
                'difficulty':difficulty_labels,
            }
        )

        return [out]
    
    #== Temporary tokenizer for truncating inputs =================================================#
    @property
    def tokenizer(self):
        if not hasattr(self, '_tokenizer'):
            from transformers import  AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        return self._tokenizer