from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoProcessor
import transformers
import torch
from pos_test import *
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace this with your own checkpoint
model_checkpoint_ner = "ner_models"
model_checkpoint_pos = "pos_models"
sentence = "My name is Sylvain and I work at Hugging Face in Brooklyn."

def ner_test(sentence, model_checkpoint):

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # processor = AutoProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint).to(device)

    # Tokenize input sentence
    tokens = tokenizer(sentence, return_offsets_mapping=True, truncation=True, return_tensors="pt").to(device)
    # encoded_inputs = processor(tokens, return_tensors="pt", truncation=True)

    # Run model on tokenized input
    with torch.no_grad():
        logits = model(tokens.input_ids, attention_mask=tokens.attention_mask)[0]

    # Decode predicted labels
    predicted_labels = torch.argmax(logits, dim=-1)
    labels = [model.config.id2label[label_id] for label_id in predicted_labels[0].tolist()]

    # Match predicted labels with input tokens and offsets
    offsets = tokens.offset_mapping[0].tolist()
    entities = []
    current_entity = None
    for i, label in enumerate(labels):
        if label == "O":
            # If the current token has no label, we are not inside an entity
            if current_entity is not None:
                # If we were inside an entity, append it to the list of entities
                entities.append(current_entity)
                current_entity = None
        else:
            # If the current token has a label, we are inside an entity
            entity_type = label[2:] # Remove the "B-" or "I-" prefix from the label
            entity_text = sentence[offsets[i][0]:offsets[i][1]]
            if current_entity is None:
                # If we were not inside an entity, start a new entity
                current_entity = (entity_type, entity_text)
            elif current_entity[0] == entity_type:
                # If we are inside an entity of the same type, append the text to the current entity
                current_entity = (entity_type, current_entity[1] + "" + entity_text)
            else:
                # If we are inside an entity of a different type, close the current entity and start a new one
                entities.append(current_entity)
                current_entity = (entity_type, entity_text)

    # If we were inside an entity at the end of the loop, append it to the list of entities
    if current_entity is not None:
        entities.append(current_entity)

    return dict(entities)

# entities = ner_test(sentence, model_checkpoint_ner)
# print(entities)
# print(type(list(entities.keys())[0]))