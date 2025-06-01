# This script runs locally (w/LM Studio server) and an embedding model loaded.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, 'C:\\Users\\nseda\\Documents\\GitHub\\LLM-SQL-Retrieval')
from server.config import *
import json
import re
from server.keys import *


document_to_embed = "knowledge\\table_descriptions.txt"

def get_embedding(text, model=embedding_model):
   text = text.replace("\n", " ")
   return local_client.embeddings.create(input = [text], model=model).data[0].embedding

# Read the text document
with open(document_to_embed, 'r', encoding='utf-8', errors='ignore') as infile:
    text_file = infile.read()

# A new strategy for chunking, parsing the txt file with regex
pattern = r"Table:\s*(.*?)\s*Description:\s*(.*?)(?=Table:|\Z)"
matches = re.findall(pattern, text_file, re.DOTALL)

chunks = []
for table_name, description in matches:
    chunks.append({
        "name": table_name.strip(),
        "content": description.strip()
    })
        
# Create the embeddings
embeddings = []
for i, chunk in enumerate(chunks):
    print(f'{i + 1} / {len(chunks)}')
    vector = get_embedding(chunk['content'])
    embeddings.append({
        'name': chunk['name'],
        'content': chunk['content'],
        'vector': vector
    })

# Save the embeddings to a json file
output_filename = os.path.splitext(document_to_embed)[0]
output_path = f"{output_filename}.json"

with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(embeddings, outfile, indent=2, ensure_ascii=False)

print(f"Finished vectorizing. Created {document_to_embed}")