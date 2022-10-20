import os, time, shutil
import faiss
import numpy as np

import clip
import torch
from PIL import Image
from clip_onnx import clip_onnx
        
DATABASE_ROOT_DIR   = '/data/prompt-search-imgs2/'

IMG_FILES_EXTENSION = ".webp"
TXT_FILES_EXTENSION = ".txt"

DEVICE   = "cpu"
USE_CLIP = False

#################################################################################################

print("Loading CLIP database search tools...")

IMAGES_DIR  = os.path.join(DATABASE_ROOT_DIR, "images")
INDICES_DIR = os.path.join(DATABASE_ROOT_DIR, "knn_indices")
ONNX_DIR    = os.path.join(DATABASE_ROOT_DIR, "clip_onnx_models")

VISUAL_INDEX_FILE_PATH  = os.path.join(INDICES_DIR, "visual_prompts.index")
TEXTUAL_INDEX_FILE_PATH = os.path.join(INDICES_DIR, "text_prompts.index")

VISUAL_EMBEDDINGS_DIR  = os.path.join(DATABASE_ROOT_DIR, "visual_embeddings")
TEXTUAL_EMBEDDINGS_DIR = os.path.join(DATABASE_ROOT_DIR, "textual_embeddings")

#PROMPTS_FROM_GENERATION_ID_DIR = os.path.join(TEXTUAL_EMBEDDINGS_DIR, "prompts_from_generation_id")
PROMPTS_FROM_PROMPT_ID_DIR     = os.path.join(TEXTUAL_EMBEDDINGS_DIR, "prompts_from_prompt_id")

VISUAL_IDS     = np.load(os.path.join(VISUAL_EMBEDDINGS_DIR,  "visual_ids.npy"))
PROMPT_IDS     = np.load(os.path.join(TEXTUAL_EMBEDDINGS_DIR, "prompt_ids.npy"))
GENERATION_IDS = np.load(os.path.join(TEXTUAL_EMBEDDINGS_DIR, "generation_ids.npy"))

"""
10'745'619 visual_ids and 10'778'778 prompt_ids.
10'745'619 unique visual_ids and 1'937'885 unique prompt_ids.

Need a mapping from visual_ids to prompt_ids.
"""


# Load faiss database index for both img-search and txt-search:
VISUAL_INDEX = faiss.read_index(VISUAL_INDEX_FILE_PATH,
    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,)

TEXTUAL_INDEX = faiss.read_index(TEXTUAL_INDEX_FILE_PATH,
    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,)

CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE, jit=False)

if not USE_CLIP:
    # free memory of clip model:
    del CLIP_MODEL
    CLIP_MODEL = None
    torch.cuda.empty_cache()

ONNX_MODEL = clip_onnx(None)
ONNX_MODEL.load_onnx(
    visual_path= os.path.join(ONNX_DIR, "visual.onnx"),
    textual_path= os.path.join(ONNX_DIR, "textual.onnx"),
    logit_scale=100.0000,
)
ONNX_MODEL.start_sessions(providers=["CPUExecutionProvider"], )


######################################################################################################

def get_prompts_from_filepaths(txt_filepaths):
    prompts = []
    for filepath in txt_filepaths:
        with open(filepath, "r") as f:
            prompts.append(f.read())
    return prompts

####### CLIP #########

def get_imgs_from_img_clip(processed_img, loaded_index, top_k, verbose=0):
    with torch.no_grad():
        visual_embedding = CLIP_MODEL.encode_image(processed_img, )
        visual_embedding /= visual_embedding.norm(dim=-1, keepdim=True)
        visual_embedding = visual_embedding.cpu().numpy().astype('float32')

    distances, I = loaded_index.search(visual_embedding, top_k)
    top_k_img_paths = [os.path.join(IMAGES_DIR, str(VISUAL_IDS[idx]) + IMG_FILES_EXTENSION) for idx in I[0]]
    return top_k_img_paths

def get_imgs_from_txt_clip(tokenized_text, loaded_index, top_k, verbose=0):
    with torch.no_grad():
        text_embedding = CLIP_MODEL.encode_text(tokenized_text, )
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy().astype('float32')

    distances, I = loaded_index.search(text_embedding, top_k)
    top_k_img_paths = [os.path.join(IMAGES_DIR, str(VISUAL_IDS[idx]) + IMG_FILES_EXTENSION) for idx in I[0]]
    return top_k_img_paths

####### ONNX #########

def filter_duplicates(item_list, n):
    """
    Return the first n, non-duplicate items from item_list
    Also return the corresponding indices
    """
    filtered_item_list, indices = [], []
    for i, item in enumerate(item_list):
        if item not in filtered_item_list:
            filtered_item_list.append(item)
            indices.append(i)
        if len(filtered_item_list) == n:
            break
    return filtered_item_list, indices

def map_visual_ids_to_textual_ids(visual_ids):
    prompt_ids = []

    for visual_id in visual_ids:
        # find the index of visual_id in the textual GENERATION_IDS:
        idx = np.where(GENERATION_IDS == visual_id)[0][0]

        # use that index to get the prompt_id from the textual PROMPT_IDS:
        prompt_id = PROMPT_IDS[idx]
        prompt_ids.append(prompt_id)

    return prompt_ids

def make_square(img):
    # pad the PIL img with black pixels to make it square:
    w = max(img.size)
    canvas = np.zeros((w, w, 3), dtype=np.uint8)
    canvas[:img.size[1], :img.size[0], :] = np.array(img)
    return Image.fromarray(canvas)



def find_top_k_matches(input_data, mode, top_k, verbose=0, make_img_square=True):
    """
    performs fast similarity search on the database_index
    mode = ['img2img', 'txt2img', 'img2txt', 'txt2txt']

    if in_mode is "img", input_data is a PIL image
    if in_mode is "txt", input_data is a string

    if out_mode is "img", returns a list of top_k image paths
    if out_mode is "txt", returns a list of top_k text prompts  

    if mode is "img2img2txt", first map img2img and then return prompts that generated those imgs

    """
    mode_types = mode.split('2')
    if len(mode_types) == 2:
        in_type, out_type = mode_types
        out_type2 = None
    elif len(mode_types) == 3:
        in_type, out_type, out_type2 = mode_types

    if in_type == 'img':
        if make_img_square:
            input_data = make_square(input_data)
        input_data     = CLIP_PREPROCESS(input_data).unsqueeze(0).to(DEVICE)
        embedding      = ONNX_MODEL.encode_image(input_data.numpy(), )
    elif in_type == 'txt':
        input_data     = clip.tokenize([input_data], truncate=True,).to(DEVICE)
        embedding      = ONNX_MODEL.encode_text(input_data.long().numpy(), )

    embedding /= np.linalg.norm(embedding, axis=-1, keepdims=True)

    if out_type == "img":
        database_index = VISUAL_INDEX
    elif out_type == "txt":
        database_index = TEXTUAL_INDEX

    # Fine closest k matches in the database: (20x cause we're expecting duplicates which will be filtered later)
    distances, db_indices = database_index.search(embedding, top_k*20)
    distances, db_indices = distances[0], db_indices[0]

    if out_type == 'img':
        top_k_img_paths = [os.path.join(IMAGES_DIR, str(VISUAL_IDS[idx]) + IMG_FILES_EXTENSION) for idx in db_indices]
        return_list     = top_k_img_paths
    elif out_type == 'txt':
        top_k_prompt_paths = [os.path.join(PROMPTS_FROM_PROMPT_ID_DIR, str(PROMPT_IDS[idx]) + TXT_FILES_EXTENSION) for idx in db_indices]
        return_list        = get_prompts_from_filepaths(top_k_prompt_paths)

    return_list, indices = filter_duplicates(return_list, top_k)
    distances  = [distances[i]  for i in indices]
    db_indices = [db_indices[i] for i in indices]

    if out_type2 is not None: # img2img2txt mode: return prompts corresponding to most similar images
        visual_ids = [str(VISUAL_IDS[idx]) for idx in db_indices]
        # Map visual ids onto prompts:
        textual_ids = map_visual_ids_to_textual_ids(visual_ids)
        top_k_prompt_paths = [os.path.join(PROMPTS_FROM_PROMPT_ID_DIR, str(textual_id) + TXT_FILES_EXTENSION) for textual_id in textual_ids]
        return_list        = get_prompts_from_filepaths(top_k_prompt_paths)

            
    if verbose:
        print(f"\nTop-{top_k} matching {out_type} for query {in_type}:")
        for i, el in enumerate(return_list):
            print(f"{i:02}: {distances[i]:.4f} --> {el}")

    return return_list, distances


def save_imgs_to_dir(img_paths, scores, save_dir, subdir):
    """
    Load all the imgs, convert them to .jpg and save them to a directory.
    """
    os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        score = scores[i]
        img.save(os.path.join(save_dir, subdir, f"{i:02}_{score:.3f}.jpg"))

#####################################################################################################

if __name__ == "__main__":
    #from clip_search import *

    input_img_path = "query_image.png"
    input_img_path = "earthrise.jpg"
    input_prompt   = "a glass museum, filled with thousands of neurons, mystical atmosphere"
    top_k          = 5

    input_img  = Image.open(input_img_path)
    start_time = time.time()
    img2img_matches, img2img_similarities = find_top_k_matches(input_img,    "img2img", top_k, verbose=1)
    txt2img_matches, txt2img_similarities = find_top_k_matches(input_prompt, "txt2img", top_k, verbose=1)
    txt2txt_matches, txt2txt_similarities = find_top_k_matches(input_prompt, "txt2txt", top_k, verbose=1)
    img2txt_matches, img2txt_similarities = find_top_k_matches(input_img,    "img2txt", top_k, verbose=1)
    img2img2txt_matches, img2img2txt_sims = find_top_k_matches(input_img,"img2img2txt", top_k, verbose=1)

    print(f"\n ---> All 4 ONXX-DB searches completed in {time.time() - start_time:.3f} seconds",)

    output_dir = "search-results"

    try:
        shutil.rmtree(output_dir)
    except Exception as e:
        pass
    os.makedirs(output_dir, exist_ok = True)

    save_imgs_to_dir(img2img_matches, img2img_similarities, output_dir, "img2img")
    save_imgs_to_dir(txt2img_matches, txt2img_similarities, output_dir, "txt2img")
