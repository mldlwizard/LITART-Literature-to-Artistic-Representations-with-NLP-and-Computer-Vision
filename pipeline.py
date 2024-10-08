import spacy
import neuralcoref
import stanza
import torch
from ner_test import *
from coref_dependency import *
# from summarize import *
# from stable_diffusion import *
from tqdm import tqdm
import json
from checks import *
import nltk
nltk.download('punkt')

# NOTE: Change the path of book if you want to get results for a certain book
# That book must contain the character list of that book. For that, run character_list.py.

# Path to all characters JSON w.r.t book/novel/story/poem
# path_to_chars = "/home/kambhamettu.s/Visualizing Novel Characters/character_contents.json"
path_to_chars = "/home/kambhamettu.s/Visualizing Novel Characters/short_story_contents.json"
# Set the desired book's path
path_to_book = "/home/kambhamettu.s/Visualizing Novel Characters/new_books/short_story.txt"
# Set the path to save all characters of a book's json
path_to_book_desc= "/home/kambhamettu.s/Visualizing Novel Characters/book_descs"
# The path to all summaries
path_to_book_summary = "/home/kambhamettu.s/Visualizing Novel Characters/book_summaries"

# Read the book
with open(path_to_book, 'r', encoding='utf-8') as file:
    # Read the contents of the file
    contents = file.read()

# Read the characters of the book
with open(path_to_chars, 'r') as j:
    character_contents = json.loads(j.read())

# Fetch the characters pertaining to the book
character_list = character_contents["short_story"]

# Perform coreference resolution on the contents of the book to replace pronouns.
coref = coref_resolution(contents)

# Split the text into sentences
sentences = nltk.sent_tokenize(coref)

# Extract descriptions
for character in character_list:
    character_desc = []
    normal_desc = []
    print(f"Extracting descriptions for {character}")
    for sentence in tqdm(sentences):
        # NER Tagging
        entities = ner_test(sentence, model_checkpoint_ner)
        try:
            # Filter only character relevant sentences
            if entities['PER'] == character or entities['MISC'] == character:
                descriptive_sentence = descriptive_filter(sentence)
                # Filter the descriptive sentences only.
                if descriptive_sentence:
                    character_desc.append(descriptive_sentence)
                else:
                    normal_desc.append(sentence)
        except:
            pass
    # keep both the descriptive sentences and all sentences
    full_desc = " ".join(character_desc)
    full_desc_normal = " ".join(normal_desc)

    # Save them as json files
    with open(os.path.join(path_to_book_desc, str(character)+"_desc_.json"), 'w') as f:
        f.write(json.dumps(full_desc) + "\n")

    with open(os.path.join(path_to_book_desc, str(character)+"_normal_.json"), 'w') as f:
        f.write(json.dumps(full_desc_normal) + "\n")

print("All Descriptions Extracted!!")

# # Summarizer
# print("Begin Abstractive Summarizing")
# for char_desc in os.listdir(path_to_book_desc):
#     char_path = os.path.join(path_to_book_desc, char_desc)
#     description = get_character_summary(char_path)

#     ARTICLE_TO_SUMMARIZE = description
#     inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")

#     # Global attention on the first token (cf. Beltagy et al. 2020)
#     global_attention_mask = torch.zeros_like(inputs)
#     global_attention_mask[:, 0] = 1

#     # Generate Summary
#     summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=80)
#     abstractive_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

#     # Dump the summary
#     with open(os.path.join(path_to_book_summary, str(character)+"_normal_.json"), 'w') as f:
#         f.write(json.dumps(full_desc_normal) + "\n")

# # Text-to-Image Generation
# print("Begin Text-to-Image Generation")

# for summary in os.listdir(path_to_book_summary):
#     with open(os.path.join(path_to_book_summary, summary), 'r', encoding='utf-8') as file:
#         # Read the contents of the file
#         prompt = file.read()

#     # Generate 5 images from runwayml
#     for i in range(5):
#         runway_pipe(prompt).images[0].save(f"results/runawayml{i}.png")

#     # Generate 5 images from stability ai
#     for i in range(5):
#         pipe(prompt).images[0].save(f"results/stabilityAI{i}.png")