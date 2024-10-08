
#----------------------- Import Libraries --------------------------#

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
# from huggingface_hub import Repository, get_full_repo_name

from huggingface_hub import notebook_login

import pandas as pd
import numpy as np

from token_classification_functions import *

#----------------------- Load Dataset ---------------------------#

class_task = "ner"

# CONL 2003 Dataset
raw_datasets = load_dataset("conll2003")
print(raw_datasets)


# Extract NER Features and Label Names
token_feature = raw_datasets["train"].features[class_task + '_tags']
label_names = token_feature.feature.names
print("\nLabel Names: ", label_names)

'''
O means the word doesn’t correspond to any entity.
B-PER/I-PER means the word corresponds to the beginning of/is inside a person entity.
B-ORG/I-ORG means the word corresponds to the beginning of/is inside an organization entity.
B-LOC/I-LOC means the word corresponds to the beginning of/is inside a location entity.
B-MISC/I-MISC means the word corresponds to the beginning of/is inside a miscellaneous entity.
'''

#--------------------------- Preprocessing Data -------------------------#

# Initializing Pretrained Tokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# Tokenize the dataset and align the labels correspondingly
# Eg: If a word is split into 2 tokens, the labels would also have to be printed twice
tokenized_datasets = raw_datasets.map(lambda examples: tokenize_and_align_labels(examples, tokenizer,class_task),
                                        batched=True,
                                        remove_columns=raw_datasets["train"].column_names)

# Create 2 dicts for further use
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


# Initialize Data Collator (huggingface) for padding, etc
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# Creating Dataloaders
train_dataloader = DataLoader(tokenized_datasets["train"],
                                shuffle=True,
                                collate_fn=data_collator,
                                batch_size=8,)

eval_dataloader = DataLoader(tokenized_datasets["validation"], 
                                collate_fn=data_collator, 
                                batch_size=8)


#--------------- Example of Sentence and corresponding Output ----------------------#

filename = "token_classification_output_example.py"
exec(compile(open(filename, "rb").read(), filename, 'exec'))

#--------------------- Define Model and Training Parameters --------------------------#

# Epochs
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

# Directory to save models
output_dir = class_task + "_models"


# Model
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
                                                        id2label=id2label,
                                                        label2id=label2id,)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Prepare Accelerator
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# LR Scheduler
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# Training
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered, label_names)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    # Save the models
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
















