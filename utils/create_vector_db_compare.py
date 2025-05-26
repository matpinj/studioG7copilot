import sys
sys.path.insert(0, 'C:\\Users\\Matea\\Documents\\IAAC\\3\\studio\\SQL\\LLM-SQL-Retrieval')
from server.config import *
import json
import os
import re

document_to_embed = "knowledge\\compare_results.txt"

def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    return local_client.embeddings.create(input=[text], model=model).data[0].embedding

# Read the text document
with open(document_to_embed, 'r', encoding='utf-8', errors='ignore') as infile:
    text_file = infile.read()

# Chunk by "Level X:" or "Level building_total:"
pattern = r"Level\s+(.+?):\n\s+Matching activities: (.*?)\n\s+Not possible activities: (.*?)\n"
matches = re.findall(pattern, text_file, re.DOTALL)

chunks = []
for level, matching, not_possible in matches:
    content = f"Level {level.strip()}:\nMatching activities: {matching.strip()}\nNot possible activities: {not_possible.strip()}"
    chunks.append({
        "name": f"level_{level.strip()}",
        "content": content
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
output_path = f"{output_filename}_compare.json"

with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(embeddings, outfile, indent=2, ensure_ascii=False)

print(f"Finished vectorizing. Created {output_path}")