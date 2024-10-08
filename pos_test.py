
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Replace this with your own checkpoint
model_checkpoint = "pos_models"
sentence = "My name is Sylvain and I work at Hugging Face in Brooklyn."

def pos_testing(sentence, model_checkpoint):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

    # Tokenize input sentence
    tokens = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt")

    # Run model on tokenized input
    with torch.no_grad():
        logits = model(tokens.input_ids, attention_mask=tokens.attention_mask)[0]

    # Decode predicted labels
    predicted_labels = torch.argmax(logits, dim=-1)
    labels = [model.config.id2label[label_id] for label_id in predicted_labels[0].tolist()]

    # Match predicted labels with input tokens and offsets
    offsets = tokens.offset_mapping[0].tolist()
    pos_tags = []
    current_pos_tag = None
    for i, label in enumerate(labels):
        pos_tag = label.split("-")[-1] # Extract the POS tag from the label
        if current_pos_tag is None:
            # If we were not inside a POS tag, start a new one
            current_pos_tag = pos_tag
            current_pos_text = sentence[offsets[i][0]:offsets[i][1]]
        elif current_pos_tag == pos_tag:
            # If we are inside a POS tag of the same type, append the text to the current POS tag
            current_pos_text += "" + sentence[offsets[i][0]:offsets[i][1]]
        else:
            # If we are inside a POS tag of a different type, close the current POS tag and start a new one
            pos_tags.append(current_pos_tag)
            current_pos_tag = pos_tag
            current_pos_text = sentence[offsets[i][0]:offsets[i][1]]

    # If we were inside a POS tag at the end of the loop, append it to the list of POS tags
    if current_pos_tag is not None:
        pos_tags.append(current_pos_tag)

    return pos_tags

pos_sent = pos_testing(sentence, model_checkpoint)

# print(pos_sent)


