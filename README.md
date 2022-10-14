# causal_probing

# Setup

Setup Conda environment
```
conda create -n ma pyton=3.9
conda activate ma
pip install -r requirements
```

Install haystack
```
git clone https://github.com/deepset-ai/haystack.git
cd haystack
pip install --upgrade pip
pip install -e '.[all-gpu]'
```