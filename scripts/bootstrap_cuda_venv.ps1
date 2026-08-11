# CUDA runtime bootstrap for Windows (Python 3.13 recommended)
# 1) py -3.13 -m venv .venv-cuda
# 2) .\.venv-cuda\Scripts\activate
# 3) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 4) pip install -r requirements.txt
# 5) copy .env.example .env  (QDRANT_PREFER_LOCAL=true if Docker missing)
