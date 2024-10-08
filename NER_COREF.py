from transformers import BertTokenizer, BertForTokenClassification, AutoTokenizer, AutoModel, AutoModelForTokenClassification
from transformers import pipeline

# Load NER model
ner_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Load coreference resolution model
coref_tokenizer = AutoTokenizer.from_pretrained("shtoshni/spanbert_coreference_large")
coref_model = AutoModel.from_pretrained("shtoshni/spanbert_coreference_large")


# Define input text
input_text = "John Smith works at Apple. He loves it there."

# Extract named entities
ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
ner_output = ner(input_text)

# Extract sentences
sentences = input_text.split(".")

# Resolve coreferences
coref = pipeline("coref", model=coref_model, tokenizer=coref_tokenizer)
coref_output = coref(sentences)

# Print output
for cluster in coref_output:
    mentions = cluster['mentions']
    rep_mention = mentions[0]  # representative mention
    for mention in mentions:
        if mention['text'] == rep_mention['text']:
            continue
        start = mention['start']
        end = mention['end']
        sentence_idx = mention['sent_num']
        sentence = sentences[sentence_idx].strip()
        print(f"{mention['text']} refers to {rep_mention['text']} in sentence: '{sentence}'")
