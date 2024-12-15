import openai
import os
import time
import numpy as np

from types import SimpleNamespace

openai.organization = "OPENAI_API_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatgptSystem:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def text_response(self, input_text):
        completion = openai.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": input_text,
                },
            ],
            max_tokens = 1,
            seed = 1,
            logprobs = True,
            top_logprobs = 20,
        )
        return None

class ClassifierChatgptsystem(ChatgptSystem):
    def __init__(self, decoder_prefix=''):
        super().__init__() 
        self.set_up_prompt_classifier()
        self.decoder_prefix=decoder_prefix
        
    def set_up_prompt_classifier(self):
        # Set up label words
        self.label_words = ['A', 'B']

    def prompt_classifier_response(self, input_text):
        self.set_up_prompt_classifier()
        
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": input_text + self.decoder_prefix,
                },
            ],
            max_tokens=1,
            seed=1,
            logprobs=True,
            top_logprobs=20
        )

        token_probs = completion.choices[0].logprobs.content[0].top_logprobs
        logprobs = [-100]*len(self.label_words)
        
        #for item in token_probs:
        #    print(item.token, item.logprob)
        
        for item in token_probs:
            if item.token == self.label_words[0]: logprobs[0] = item.logprob
            if item.token == self.label_words[1]: logprobs[1] = item.logprob
        
        if min(logprobs) == -100:
            print('eh not great')
            
        output_text = 'A' if logprobs[0] > logprobs[1] else 'B'
        
        return SimpleNamespace(
            output_text=output_text, 
            logits=[float(p) for p in logprobs],
            raw_probs=[float(np.exp(p)) for p in logprobs]
        )
   

class ChatGptInterfaceOld:
    def __init__(self):
        pass

    @classmethod
    def text_response(cls, input_text, top_k:int=None, do_sample:bool=False, max_new_tokens:int=None, **kwargs):
        temp = 1 if do_sample else 0
        k = 0
        while k < 3:
            try:
                output = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-0301',
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=temp, # 0.0 = deterministic
                    max_tokens=max_new_tokens, # max_tokens is the generated one,
                )
                output_text = output.choices[0].message.content
                return SimpleNamespace(output_text=output_text)
            except:
                time.sleep(5)
                k += 1
        raise Exception

