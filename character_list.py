from coref_dependency import *
from checks import *
import os

# NOTE: change the dir to book you want.

# path_to_books = "/home/nippani.a/Visualizing Novel Characters/new_books"
path_to_book_desc= "/home/kambhamettu.s/Visualizing Novel Characters/book_descs"
path_to_chars = "/home/kambhamettu.s/Visualizing Novel Characters/short_story_contents.json"
path_to_story = "/home/kambhamettu.s/Visualizing Novel Characters/new_books/short_story.txt"

# change the name of your book with its respective .txt file name
contents = get_book_contents(path_to_book_desc, path_to_story, "short_story.txt")

if contents:
    # Coreference Resolution
    # Replace pronouns with the entities they are referring to
    coref = coref_resolution(contents)

    # Tokenize into sentences
    print("Tokenizing into sentences")
    sentences = nltk.sent_tokenize(coref)

    # Character Extraction
    characters, character_flag = get_character_contents(path_to_chars, "short_story")
    print(characters, character_flag)
    if character_flag == False:
        characters = character_extraction(sentences, "short_story", path_to_chars)
    
    character_descriptions = extract_description(sentences, characters, str(book[:-4]), path_to_book_desc)

    print("Ya DONE!")