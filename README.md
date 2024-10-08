# LITART: Literature to Artistic Representations with NLP and Computer Vision

## Project Description
LITART is a project developed under the guidance of Dr. Pavlu Virgil, aimed at transforming textual descriptions of literary characters into visual representations using Natural Language Processing (NLP) and Computer Vision (CV) techniques. The project focuses on analyzing the physical attributes of characters described in books through NLP techniques such as Named Entity Recognition (NER), Part-of-Speech (POS) tagging, dependency parsing, and coreference resolution. The extracted data is then used to fine-tune text-to-image generative models to create visual representations of these literary characters. 

By leveraging advanced NLP and CV technologies, LITART seeks to bridge the gap between textual storytelling and visual depictions, offering a new medium for understanding and experiencing literary characters.

## Code Functionality of All Scripts

### `gutenbergData.py`
- **Purpose**: This script processes raw text files from Project Gutenberg, extracting relevant character descriptions and physical traits using various NLP techniques. It helps prepare the data needed for downstream tasks, such as generating images.
- **Key Functions**:
    - **Text Parsing**: Splits the input text into manageable sections for further NLP processing.
    - **NER and POS Tagging**: Applies Named Entity Recognition and Part-of-Speech tagging to identify characters and their attributes.
    - **Coreference Resolution**: Links references to the same characters across the text, ensuring accurate data extraction.

### `character_list.py`
- **Purpose**: Extracts and compiles a list of characters from a provided text file, identifying them using NLP techniques such as NER.
- **Key Functions**:
    - **Character Identification**: Uses NER to find character mentions in the text.
    - **List Compilation**: Generates a comprehensive list of characters to be used in the pipeline for further processing.

### `pipeline.py`
- **Purpose**: Orchestrates the entire process from character extraction to image generation, leveraging NLP techniques and a fine-tuned text-to-image model.
- **Key Functions**:
    - **Data Flow Management**: Coordinates the output of character extraction and feeds it into the image generation model.
    - **Model Fine-Tuning**: Fine-tunes the text-to-image generative model based on the extracted character descriptions.

## Folder Structure

### `new_books/`
- **Description**: This folder contains raw `.txt` files of books that you want to process for character description and visualization.
- **Usage**: Place the text files of new books in this directory before running the `character_list.py` and `pipeline.py` scripts.

### `pre_existing_books/`
- **Description**: Stores preprocessed character lists from existing books, allowing users to experiment with pre-existing data without having to rerun the character extraction process.
- **Usage**: If you are experimenting with existing character lists, use this directory's files as input for the `pipeline.py` script.

### `results/`
- **Description**: After running the scripts, the output files, including character lists and generated images, are saved in this directory.
- **Usage**: Check this folder for the final output of the visual representations and the processed character lists.

### `models/`
- **Description**: This folder contains the models used or fine-tuned during the image generation process.
- **Usage**: The text-to-image models required for generating the visual representations are stored here. You can replace or update models as needed.

### `utils/`
- **Description**: This folder holds utility scripts and helper functions that support the primary NLP and CV tasks.
- **Usage**: Includes helper functions for data loading, processing, and model fine-tuning. 

## CLI Commands to Run the Code

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/mldlwizard/LITART-Literature-to-Artistic-Representations-with-NLP-and-Computer-Vision.git
    cd LITART-Literature-to-Artistic-Representations-with-NLP-and-Computer-Vision
    ```

2. **For Using a New Book**:
    1. Upload the `.txt` file to the `new_books` directory.
    2. Run the `character_list.py` script with the appropriate paths set:
       ```bash
       python3 character_list.py --input_path new_books/<book_filename>.txt --output_path results/
       ```
    3. Run the `pipeline.py` script:
       ```bash
       python3 pipeline.py --input_path results/<character_list_filename>.json --output_path results/
       ```
    4. Check the results in the `results/` directory.

3. **For Experimenting with Existing Books**:
    1. Simply run the `pipeline.py` script:
       ```bash
       python3 pipeline.py --input_path pre_existing_books/<preprocessed_filename>.json --output_path results/
       ```
    2. Check the results in the `results/` directory.

## Installation and Dependencies
Ensure you have all the required dependencies installed:
```bash
pip install -r requirements.txt
```

## Heads up
Enjoy exploring literary characters visually! Be patient with the process, as it may take time, especially with larger books.
