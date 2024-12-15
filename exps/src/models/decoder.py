import torch
import torch.nn.functional as F

from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

MODEL_URLS = {
    'llama2-7b':'meta-llama/Llama-2-7b-hf',
    'llama2-13b':'meta-llama/Llama-2-13b-hf',
    'llama2-7b-chat':'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b-chat':'meta-llama/Llama-2-13b-chat-hf',
    'llama3-8b-inst':'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistral-7b-v2':'mistralai/Mistral-7B-Instruct-v0.2'
}

class DecoderSystem:
    def __init__(self, system_name:str, device=None):
        self.system_name = system_name
        
        system_url = MODEL_URLS[system_name]
        self.tokenizer = AutoTokenizer.from_pretrained(system_url)

        # set device if not specified
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # load model in fp16 if cuda (assumed), else in cpu
        if 'cuda' in device:
            self.model = AutoModelForCausalLM.from_pretrained(system_url, torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(system_url)

        self.to(device)
        self.device = device

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def text_response(self, input_text, top_k:int=10, do_sample:bool=False, max_new_tokens:int=None):
        chat_input = [{"role": "user", "content": input_text}]
        formatted_text = self.tokenizer.apply_chat_template(chat_input, tokenize=False)
        inputs = self.tokenizer(formatted_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                top_k=top_k,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_tokens = output[0]
        
        input_tokens = inputs.input_ids[0]
        new_tokens = output_tokens[len(input_tokens):]
        assert torch.equal(output_tokens[:len(input_tokens)], input_tokens)
        
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return SimpleNamespace(output_text=output_text)
    

class ClassifierDecoderSystem(DecoderSystem):
    def __init__(self, system_name:str, decoder_prefix:str, device=None):
        super().__init__(system_name=system_name, device=device) 
        self.set_up_prompt_classifier()
        self.decoder_prefix = decoder_prefix

    def set_up_prompt_classifier(self):
        # Set up label words
        label_words = ['A', 'B']
        if self.system_name == 'llama3-8b-inst': label_words = [' A', ' B']
            
        label_ids = [self.tokenizer(word, add_special_tokens=False).input_ids for word in label_words]
        if any([len(i)>1 for i in label_ids]):
            print('warning: some label words are tokenized to multiple words')
        self.label_ids = [int(self.tokenizer(word, add_special_tokens=False).input_ids[0]) for word in label_words]
        
    def prompt_classifier_response(self, input_text):
        chat_input = [{"role": "user", "content": input_text}]
        formatted_text = self.tokenizer.apply_chat_template(chat_input, tokenize=False)
        
        formatted_text = formatted_text + f" {self.decoder_prefix}"
        
        inputs = self.tokenizer(formatted_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
        
        vocab_logits = output.logits[:,-1]
        
        #if self.system_name == 'llama3-8b-inst':
        #    self.debug_output_logits(input_text, vocab_logits)
        
        class_logits = vocab_logits[0, tuple(self.label_ids)]
        raw_class_probs = F.softmax(vocab_logits, dim=-1)[0, tuple(self.label_ids)]
        pred = int(torch.argmax(class_logits))

        pred_to_output = {0:'A', 1:'B'}
        output_text = pred_to_output[pred]

        return SimpleNamespace(
            output_text=output_text, 
            logits=[float(i) for i in class_logits],
            raw_probs=[float(i) for i in raw_class_probs]
        )
   
    def debug_output_logits(self, input_text, logits):
        # Debug function to see what outputs would be
        indices = logits.topk(k=5).indices[0]
        print('\n\n', '-'*50, "\nINPUT TEXT\n", '-'*50, '\n')
        print(input_text)
        print('\n\n', '-'*50, "\nSET LABEL IDS \n", '-'*50, '\n')
        print(self.label_ids)

        print('\n\n', '-'*50, "\nTOP K IDS \n", '-'*50, '\n')
        print(indices, '\n')
        print(logits[0, indices], '\n')
        print(self.tokenizer.decode(indices), '\n')
        print('\n\n')
        import time; time.sleep(1)

    #== Setup methods =============================================================================#
