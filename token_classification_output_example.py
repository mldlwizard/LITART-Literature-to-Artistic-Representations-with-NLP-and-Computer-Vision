
# Print an example of a sentence and corresponding NER Tags
words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0][class_task + '_tags']
print("\nTraining Input: ",words)
print("Annotated Output: ",labels)

# Better Visualization
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print("\nBetter Visualization:\n")
print(line1)
print(line2)


# Tokenizing Words
inputs = tokenizer(words, is_split_into_words=True)
word_ids = inputs.word_ids()
print("\nbert-base-case pretrained Autotokenizer tokens: \n",inputs.tokens())
print("Word Ids: \n",inputs.word_ids())




upd_labels = align_labels_with_tokens(labels, word_ids)
print("New Aligned Labels: ",upd_labels)


line1 = ""
line2 = ""
for word, label in zip(inputs.tokens(), upd_labels):
    try:
        full_label = label_names[label]
    except:
        full_label = "None"

    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print("\nBetter Visualization of New Tokens:\n")
print(line1)
print(line2)


# Print an example of tokenized output
print("\nTokenized Output: \n",tokenized_datasets["train"][0])
print("Converting IDs back to tokens: \n",tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


# Example of a batch from data collator
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
print("\nExample of working of Data Collator: \n",batch["labels"])

