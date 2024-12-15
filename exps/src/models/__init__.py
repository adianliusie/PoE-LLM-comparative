from .flant5 import ClassifierFlanT5System
from .decoder import ClassifierDecoderSystem
from .openai import ClassifierChatgptsystem

def load_interface(system:str, decoder_prefix:str, device:str):
    if system in ['flant5-base', 'flant5-large', 'flant5-xl', 'flant5-xxl']:
        interface = ClassifierFlanT5System(system, decoder_prefix, device)
    elif system in ['llama2-7b-chat', 'llama2-13b-chat', 'mistral-7b-v2', 'llama3-8b-inst']:
        interface = ClassifierDecoderSystem(system_name=system, decoder_prefix=decoder_prefix, device=device)
    elif system in ['chatgpt']:
        interface = ClassifierChatgptsystem(decoder_prefix)
    return interface