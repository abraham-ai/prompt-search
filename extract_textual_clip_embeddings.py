import os
import csv

import numpy as np
import torch
import clip

USE_CACHE = False
BATCH_SIZE = 2048
csv_data_path = '/data/prompt-search-imgs2/data.csv'
OUTDIR = "/data/prompt-search-imgs2/textual_embeddings"

os.makedirs(OUTDIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("Loading all prompts from csv...", flush = True)

with open(csv_data_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    _headers = next(reader)

    # csv headers:
    # prompt, generation_id, prompt_id
    prompt_data = set([(row[0], row[1], row[2]) for row in reader if row[2] != ''])

print(f"Found {len(prompt_data)} prompts in dataset", flush = True)
prompts       = [data[0] for data in prompt_data]
generation_id = [data[1] for data in prompt_data]
prompt_ids    = [data[2] for data in prompt_data]

# Save prompt ids:
prompt_ids_filename = os.path.join(OUTDIR, f"prompt_ids.npy")
np.save(prompt_ids_filename, prompt_ids)
print("All prompt ids saved to ", prompt_ids_filename, flush = True)

# Save generation ids:
generation_ids_filename = os.path.join(OUTDIR, f"generation_ids.npy")
np.save(generation_ids_filename, generation_id)
print("All generation ids saved to ", generation_ids_filename, flush = True)

# csv headers:
# prompt, generation_id, prompt_id
prompt_id_prompt_data        = list(set([(row[0], row[2]) for row in prompt_data]))
generation_id_to_prompt_data = list(set([(row[0], row[2]) for row in prompt_data]))

# Save prompts to disk as separate .txt files:
generation_id_dir = os.path.join(OUTDIR, "prompts_from_generation_id")
prompt_id_dir     = os.path.join(OUTDIR, "prompts_from_prompt_id")
os.makedirs(generation_id_dir, exist_ok=True)
os.makedirs(prompt_id_dir, exist_ok=True)

print(f"Saving {len(prompt_id_prompt_data)} prompt_id.txt files to disk...")
print(f"(Later saving {len(generation_id_prompt_data)} generation_id.txt files to disk)")

if 1:
    for i in range(len(prompt_id_prompt_data)):
        if i % 50000 == 0:
            print(f"Writing all prompts to disk... ({i} of {len(prompt_id_prompt_data)} done)")
        prompt_filename = os.path.join(prompt_id_dir, f"{prompt_id_prompt_data[i][1]}.txt")
        with open(prompt_filename, 'w') as f:
            f.write(prompt_id_prompt_data[i][0])

if 0:
    print(f"Saving {len(generation_id_prompt_data)} generation_id.txt files to disk...")
    for i in range(len(generation_id_prompt_data)):
        if i % 50000 == 0:
            print(f"Writing all prompts to disk... ({i} of {len(generation_id_prompt_data)} done)")
        prompt_filename = os.path.join(generation_id_dir, f"{generation_id_prompt_data[i][1]}.txt")
        with open(prompt_filename, 'w') as f:
            f.write(generation_id_prompt_data[i][0])

print("All prompts saved to separate .txt files", flush = True)


###################################################################################


print("Embedding..")
text_embeddings = None
batched_prompts = []
for idx, prompt in enumerate(prompts):
    batched_prompts.append(prompt)

    if len(batched_prompts) % BATCH_SIZE == 0 or idx == len(prompt_ids) - 1:
        print(f"processing -- {idx + 1} of {len(prompts)}")

        batch_text_embeddings_filename = os.path.join(OUTDIR, f"text_embeddings_{idx + 1}.npy")

        if os.path.exists(batch_text_embeddings_filename) and USE_CACHE:
            batch_text_embeddings = np.load(batch_text_embeddings_filename)

        else:
            with torch.no_grad():
                batched_text = clip.tokenize(
                    batched_prompts,
                    truncate=True,
                ).to(device)

                batch_text_embeddings = model.encode_text(batched_text, )
                batch_text_embeddings /= batch_text_embeddings.norm(
                    dim=-1, keepdim=True)

            batch_text_embeddings = batch_text_embeddings.cpu().numpy().astype('float32')

            if USE_CACHE:
                np.save(batch_text_embeddings_filename, batch_text_embeddings)

        if text_embeddings is None:
            text_embeddings = batch_text_embeddings

        else:
            text_embeddings = np.concatenate((text_embeddings, batch_text_embeddings))

        print(f"text embeddings shape -- {text_embeddings.shape}")
        print("\n")

        batched_prompts = []

print(f"{len(text_embeddings)} CLIP embeddings extracted!")
text_embeddings_filename = os.path.join(OUTDIR, f"text_embeddings.npy")
np.save(text_embeddings_filename, text_embeddings)