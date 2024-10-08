import spacy
import neuralcoref
import stanza
import torch
from ner_test import *
from tqdm import tqdm
import json
from checks import *
import nltk
# nltk.download('punkt')

# # Load SpaCy
nlp = spacy.load('en')
# Add neural coref to SpaCy's pipe
neuralcoref.add_to_pipe(nlp)

def coref_resolution(text):
    """Function that executes coreference resolution on a given text"""
    print("Resolving text coreferentially")
    doc = nlp(text)
    # fetches tokens with whitespaces from spacy document
    tok_list = list(token.text_with_ws for token in doc)
    for cluster in doc._.coref_clusters:
        # get tokens from representative cluster name
        cluster_main_words = set(cluster.main.text.split(' '))
        for coref in cluster:
            if coref != cluster.main:  # if coreference element is not the representative element of that cluster
                if coref.text != cluster.main.text and bool(set(coref.text.split(' ')).intersection(cluster_main_words)) == False:
                    # if coreference element text and representative element text are not equal and none of the coreference element words are in representative element. This was done to handle nested coreference scenarios
                    tok_list[coref.start] = cluster.main.text + \
                        doc[coref.end-1].whitespace_
                    for i in range(coref.start+1, coref.end):
                        tok_list[i] = ""

    return "".join(tok_list)


# Dependency Parser
# Download the language model
stanza.download('en')

# Build a Neural Pipeline
stanza_nlp = stanza.Pipeline('en', processors = "tokenize,mwt,pos,lemma,depparse") 

def descriptive_filter(text):
    # Parse the text
    doc = stanza_nlp(text)

    descriptive_flag = False
    # Iterate over the words in the first sentence of the document
    sent_dict = doc.sentences[0].to_dict()
    for word in sent_dict:
        # Initialize an empty list for the connections
        connections = []
        # Iterate over the children of the word
        for child in sent_dict:
            if word['id'] == child['head'] and child['lemma'] == 'be':
                if word['upos'] == 'ADJ' or word['upos'] == 'NOUN':
                    descriptive_flag = True

            elif word['id'] == child['head'] and word['lemma'] == 'have':
                if word['id'] == child['head'] and child['deprel'] == 'nsubj':
                    for new_child in sent_dict:
                        if word['id'] == new_child['head'] and new_child['deprel'] == 'obj':
                            descriptive_flag = True
    if descriptive_flag:
        return text
    else:
        return descriptive_flag

def character_extraction(sentences, title, path_to_chars):
    characters = []

    # Perform NER
    # Extract 'PER' or 'MISC' tags
    for sentence in tqdm(sentences, desc=f"Extracting Characters of {title}"):
        entities = ner_test(sentence, model_checkpoint_ner)
        try:
            if entities['PER'] != None or entities['MISC'] != None:
                characters.append(entities['PER'])
                characters.append(entities['MISC'])
        except:
            pass
    char_dict = {title: list(set(characters))}

    # Save Entities Extracted
    with open(path_to_chars, "w") as outfile:
        json.dump(char_dict, outfile)

    return characters


# Description extraction
def extract_description(sentences, characters, title, path_to_book_desc):
    book_data = {"book_title": title, "character_name": None, "description": None}
    print("Extracting Character Descriptions")

    all_descriptions = []

    for char_count, character in enumerate(characters):
        character_desc = []
        normal_desc = []
        temp_data = book_data.copy()
        temp_data["character_name"] = character

        for sentence in tqdm(sentences, desc=f"Description of  {character} [{char_count}/{len(characters)}]"):
            entities = ner_test(sentence, model_checkpoint_ner)
            try:
                if entities['PER'].lower() == character or entities['MISC'] == character:
                    print("here")
                    descriptive_sentence = descriptive_filter(sentence)
                    print(descriptive_sentence)
                    if descriptive_sentence:
                        character_desc.append(descriptive_sentence)
                    else:
                        normal_desc.append(sentence)
            except:
                pass

        # If no descriptive sentences are found for a character,
        # Then use all the sentences the character has been mentioned as character description
        if len(character_desc) > 0:
            temp_data["description"] = "".join(character_desc)
        else:
            temp_data["description"] = "".join(normal_desc)

        print(temp_data)
        all_descriptions.append(temp_data)

    # Save the book descriptions as json
    with open(os.path.join(path_to_book_desc, str(title + ".json")), 'w') as f:
        for item in character_desc:
            f.write(json.dumps(item) + "\n")

    return all_descriptions

# import spacy
# import neuralcoref
# import pandas as pd 
# import itertools
# import opennre
# from ner import *

# # Load SpaCy
# nlp = spacy.load('en')
# # Add neural coref to SpaCy's pipe
# neuralcoref.add_to_pipe(nlp)

# def coref_resolution(text):
#     """Function that executes coreference resolution on a given text"""
#     doc = nlp(text)
#     # fetches tokens with whitespaces from spacy document
#     tok_list = list(token.text_with_ws for token in doc)
#     for cluster in doc._.coref_clusters:
#         # get tokens from representative cluster name
#         cluster_main_words = set(cluster.main.text.split(' '))
#         for coref in cluster:
#             if coref != cluster.main:  # if coreference element is not the representative element of that cluster
#                 if coref.text != cluster.main.text and bool(set(coref.text.split(' ')).intersection(cluster_main_words)) == False:
#                     # if coreference element text and representative element text are not equal and none of the coreference element words are in representative element. This was done to handle nested coreference scenarios
#                     tok_list[coref.start] = cluster.main.text + \
#                         doc[coref.end-1].whitespace_
#                     for i in range(coref.start+1, coref.end):
#                         tok_list[i] = ""

#     return "".join(tok_list)

# text_path = '/home/kambhamettu.s/assignments/nlp/alice.txt'
# with open(text_path) as f:
#     text = f.readlines()

# new_text = "Elon Musk is a business magnate, industrial designer, and engineer. He is the founder, CEO, CTO, and chief designer of SpaceX. He is also early investor, CEO, and product architect of Tesla, Inc. He is also the founder of The Boring Company and the co-founder of Neuralink. A centibillionaire, Musk became the richest person in the world in January 2021, with an estimated net worth of $185 billion at the time, surpassing Jeff Bezos. Musk was born to a Canadian mother and South African father and raised in Pretoria, South Africa. He briefly attended the University of Pretoria before moving to Canada aged 17 to attend Queen's University. He transferred to the University of Pennsylvania two years later, where he received dual bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but decided instead to pursue a business career. He went on co-founding a web software company Zip2 with his brother Kimbal Musk."
# new_text_str = "".join(text)

# # Call the coref_resolution function with the string input
# coref_text = coref_resolution(new_text_str)
# relation_threshold = 0.8
# # entities = wikifier(coref_text, threshold=relation_threshold)

# # relation_model = opennre.get_model('wiki80_cnn_softmax')
# # relations_list = []
# # for permutation in itertools.permutations(entities, 2):
# #     for source in permutation[0]['characters']:
# #         for target in permutation[1]['characters']:
# #             # Relationship extraction with OpenNRE
# #             data = relation_model.infer(
# #                 {'text': coref_text, 'h': {'pos': [source[0], source[1] + 1]}, 't': {'pos': [target[0], target[1] + 1]}})
# #             if data[1] > relation_threshold:
# #                 relations_list.append(
# #                     {'source': permutation[0]['title'], 'target': permutation[1]['title'], 'type': data[0]})


# # relations_df = pd.DataFrame(entities)
# print(coref_text)