import numpy as np
from pathlib import Path

npz_path = Path(r"C:\Users\shami\OneDrive\Desktop\rag_project\data\embeddings.npz")
data = np.load(str(npz_path))

print(data.files)
print("keys:", list(data.keys()))
print("keys:", data.keys())

