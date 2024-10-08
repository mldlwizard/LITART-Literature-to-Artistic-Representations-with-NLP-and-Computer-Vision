import pandas as pd
import json
import os

# Fetch book contents and check if it is already parsed
def get_book_contents(path_to_book_desc, path_to_books, book):
    # check if the book is already parsed
    if os.path.isfile(os.path.join(path_to_book_desc, str(book[:-4] + ".json"))):
        print(book, " is done and dusted!")
        return False
    else:
        print("here")
        path_to_book = os.path.join(path_to_books, book)
        with open(path_to_book, 'r', encoding='utf-8') as file:
            # Read the contents of the file
            contents = file.read()
        return contents

# Fetch character list
def get_character_contents(path_to_chars, book_title):
    char_flag = False
    character_list = []

    with open(path_to_chars, 'r') as j:
        character_contents = json.loads(j.read())
    try:
        character_list = character_contents[0][book_title]
        char_flag = True
    except:
        print("Character list not available.")

    return character_list, char_flag

# Fetch Summary of character
def get_character_summary(path_to_book_summary):
    with open(path_to_book_summary, 'r') as j:
        desc_summary = json.loads(j.read())
    return desc_summary

