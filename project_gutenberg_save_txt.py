
import requests

# define the url of the book on Project Gutenberg
book_url = 'http://www.gutenberg.org/files/11/11-0.txt'

# send a GET request to the url
response = requests.get(book_url)

# check if the request was successful
if response.status_code == 200:
    # extract the text content from the response
    book_content = response.text
    # save the book content to a text file
    with open('alice_in_wonderland.txt', 'w', encoding='utf-8') as f:
        f.write(book_content)

else:
    print('Failed to retrieve book content.')
