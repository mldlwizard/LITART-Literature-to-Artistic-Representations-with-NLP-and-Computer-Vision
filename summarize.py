import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("pszemraj/led-large-book-summary")
model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/led-large-book-summary")
