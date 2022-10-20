import os

import numpy as np
from autofaiss import build_index

indices_folder = "/data/prompt-search-imgs2/knn_indices"

embeddings = np.load("/data/prompt-search-imgs2/textual_embeddings/text_embeddings.npy")
print("All text embeddings loaded, shape: ", embeddings.shape)

prompt_index_filename = os.path.join(indices_folder, "text_prompts.index")
index, index_infos = build_index(
    embeddings,
    index_path=prompt_index_filename,
)