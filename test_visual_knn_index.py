import os, time
import faiss
import numpy as np

import clip
import torch
from PIL import Image
from clip_onnx import clip_onnx
        
database_root_dir = '/data/prompt-search-imgs2/'
DEVICE = "cpu"
INPUT_IMG_PATH = "./prompt-search.png"
INPUT_PROMPT = "image of a blue robot with red background"
NUM_RESULTS = 5    
USE_ONNX = True

INDICES_FOLDER = os.path.join(database_root_dir, "knn_indices")
INDEX_FILE_PATH = os.path.join(INDICES_FOLDER, "visual_prompts.index")
VISUAL_EMBEDDINGS_DIR = os.path.join(database_root_dir, "visual_embeddings")
ONNX_DIR = os.path.join(database_root_dir, "clip_onnx_models")

prompt_index_filename = os.path.join(INDICES_FOLDER, "visual_prompts.index")
prompt_ids = np.load(os.path.join(VISUAL_EMBEDDINGS_DIR, "visual_ids.npy"))

loaded_index = faiss.read_index(
    prompt_index_filename,
    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
)


model, preprocess = clip.load("ViT-B/32", device=DEVICE)
onnx_model = clip_onnx(None)
onnx_model.load_onnx(
    visual_path= os.path.join(ONNX_DIR, "visual.onnx"),
    textual_path= os.path.join(ONNX_DIR, "textual.onnx"),
    logit_scale=100.0000,
)
onnx_model.start_sessions(providers=["CPUExecutionProvider"], )

img = Image.open(INPUT_IMG_PATH)
processed_img = preprocess(img).unsqueeze(0).to(DEVICE)




start = time.time()

with torch.no_grad():
    visual_embedding = model.encode_image(processed_img, )
    visual_embedding /= visual_embedding.norm(dim=-1, keepdim=True)
    visual_embedding = visual_embedding.cpu().numpy().astype('float32')

_, I = loaded_index.search(visual_embedding, NUM_RESULTS)

print("SIMILAR IMGS FROM INPUT IMG")
print([f"{str(prompt_ids[idx])}.jpg" for idx in I[0]])


tokenized_text = clip.tokenize([INPUT_PROMPT], truncate=True,).to(DEVICE)

with torch.no_grad():
    text_embedding = model.encode_text(tokenized_text, )
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding.cpu().numpy().astype('float32')

_, I = loaded_index.search(text_embedding, NUM_RESULTS)

print("SIMILAR IMGS FROM INPUT PROMPT")
print([f"{str(prompt_ids[idx])}.jpg" for idx in I[0]])
print(f"Regular search took {time.time() - start} seconds")

if USE_ONNX:
    print("\n########################################\n\n")
    start = time.time()

    onnx_visual_embedding = onnx_model.encode_image(processed_img.numpy(), )
    onnx_visual_embedding /= np.linalg.norm(onnx_visual_embedding, axis=-1, keepdims=True)
    onnx_visual_embedding = np.around(onnx_visual_embedding, decimals=4)
    _, I = loaded_index.search(onnx_visual_embedding, 5)
    print("ONNX SIMILAR IMGS FROM INPUT IMG")
    print([f"{str(prompt_ids[idx])}.jpg" for idx in I[0]])

    tokenized_text = clip.tokenize([INPUT_PROMPT], truncate=True,).to(DEVICE)
    onnx_text_embedding = onnx_model.encode_text(tokenized_text.long().numpy(), )
    onnx_text_embedding /= np.linalg.norm(onnx_text_embedding, axis=-1, keepdims=True)
    _, I = loaded_index.search(onnx_text_embedding, 5)
    print("ONNX SIMILAR IMGS FROM INPUT PROMPT")
    print([f"{str(prompt_ids[idx])}.jpg" for idx in I[0]])

    print(f"ONNX search took {time.time() - start} seconds")