from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, Dataset

from ..data_handler import DataHandler

import numpy as np
import evaluate

transformer_system = 'bert-base-cased'
bsz = 4
device = 'cuda:1'
    
#== load training and evaluation data ==========================
wi_train = DataHandler.load_write_and_improve('train')
wi_dev  = DataHandler.load_write_and_improve('dev')

train_data = []
for text, score in zip(wi_train[0].responses, wi_train[0].scores['cefr']):
    train_data.append(({'text':text, 'label':score}))
train_dataset = Dataset.from_list(train_data)

test_data = []
for text, score in zip(wi_dev[0].responses, wi_dev[0].scores['cefr']):
    test_data.append(({'text':text, 'label':score}))
test_dataset = Dataset.from_list(test_data)

#== Tokenize data ==============================================
tokenizer = AutoTokenizer.from_pretrained(transformer_system)
tokenize_function = lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True)
tok_train_dataset = train_dataset.map(tokenize_function, batched=True)
tok_test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = tok_train_dataset.shuffle(seed=42)
eval_dataset = tok_test_dataset

#== Select Model ===============================================
model = AutoModelForSequenceClassification.from_pretrained(transformer_system, num_labels=6)
training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=5, evaluation_strategy="epoch", report_to='none')

#== Select Evaluation Metrics ==================================
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#== Set up trainer==============================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

#== Do Training ================================================
trainer.place_model_on_device = device
trainer.train()