import os
import faiss
import numpy as np

import clip
import torch
from clip_onnx import clip_onnx


database_root_dir = '/data/prompt-search-imgs2/'
device = "cpu"

text = "This image is of an ornate fantasy cottage. A cottage with intricate stained-glass windows and doors with ivy crawling up the walls. A gingerbread style cottage straight out of a fairytale book complete with candy cane pillars and gumdrop roofs. Pentax 645"
text = "a cute cat"


INDICES_FOLDER = os.path.join(database_root_dir, "knn_indices")
#INDEX_FILE_PATH = os.path.join(INDICES_FOLDER, "visual_prompts.index")
prompt_index_filename = os.path.join(INDICES_FOLDER, "textual_prompts.index")
#VISUAL_EMBEDDINGS_DIR = os.path.join(database_root_dir, "visual_embeddings")
TEXTUAL_EMBEDDINGS_DIR = os.path.join(database_root_dir, "textual_embeddings")
ONNX_DIR = os.path.join(database_root_dir, "clip_onnx_models")

#visual_prompt_ids = np.load(os.path.join(VISUAL_EMBEDDINGS_DIR, "visual_ids.npy"))
prompt_ids = np.load(os.path.join(TEXTUAL_EMBEDDINGS_DIR, "prompt_ids.npy"))

loaded_index = faiss.read_index(
    prompt_index_filename,
    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
)

model, preprocess = clip.load("ViT-B/32", device=device)
onnx_model = clip_onnx(None)
onnx_model.load_onnx(
    visual_path= os.path.join(ONNX_DIR, "visual.onnx"),
    textual_path= os.path.join(ONNX_DIR, "textual.onnx"),
    logit_scale=100.0000,
)

onnx_model.start_sessions(providers=["CPUExecutionProvider"], )

tokenized_text = clip.tokenize(
    [text],
    truncate=True,
).to(device)

with torch.no_grad():
    text_embedding = model.encode_text(tokenized_text, )
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding.cpu().numpy().astype('float32')

tokenized_text = tokenized_text.detach().cpu().numpy().astype(np.int64)
onnx_text_embedding = onnx_model.encode_text(tokenized_text, )
# onnx_text_embedding /= onnx_text_embedding.norm(dim=-1, keepdim=True)
onnx_text_embedding = np.around(onnx_text_embedding, decimals=4)

_, I = loaded_index.search(text_embedding, 5)
print("CLIP RESULTS")
print([f"{str(prompt_ids[idx])}" for idx in I[0]])

_, I = loaded_index.search(onnx_text_embedding, 5)
print("ONNX RESULTS")
print([f"{str(prompt_ids[idx])}" for idx in I[0]])