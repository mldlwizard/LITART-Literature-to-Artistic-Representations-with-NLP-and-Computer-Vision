import pandas as pd
import numpy as np
import torch
import evaluate

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq

# Set batch size
BATCH_SIZE = 4

# Load the dataset
train_data = Dataset.from_pandas(pd.read_json('liscu_train.jsonl', lines=True))
val_data = Dataset.from_pandas(pd.read_json('liscu_val.jsonl', lines=True))
test_data = Dataset.from_pandas(pd.read_json('liscu_test.jsonl', lines=True))

# Define the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('pszemraj/led-large-book-summary')
model = AutoModelForSeq2SeqLM.from_pretrained('pszemraj/led-large-book-summary', return_dict=True)
# model.to('cuda')

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./output_dir',  # add this line
    predict_with_generate=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=10,
    learning_rate=1e-2,
    save_total_limit=1,
    save_steps=100,
    push_to_hub=False,
    logging_steps=100,
    logging_first_step=True,
    overwrite_output_dir=True,
    warmup_steps=100,
    fp16=True
)

# Define the data preprocessing function
def preprocess_function(examples):
    inputs = [f"summarize: {examples['summary'][idx]} </s> </s> character_name: {examples['character_name'][idx]} </s> </s> book_title: {examples['book_title'][idx]} </s> </s>" for idx in range(len(examples['book_title']))]
    targets = [examples['description'][idx] for idx in range(len(examples['book_title']))]
    model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding=True, truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the data
train_dataset = train_data.map(preprocess_function, batched=True, batch_size=BATCH_SIZE)
val_dataset = val_data.map(preprocess_function, batched=True, batch_size=BATCH_SIZE)
test_dataset = test_data.map(preprocess_function, batched=True, batch_size=BATCH_SIZE)

# Define the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define the evaluation metric
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Define the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(test_dataset)
print(results)
