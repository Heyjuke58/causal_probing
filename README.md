# causal_probing

## Setup

Setup Conda environment

```

conda create -n ma pyton=3.9
conda activate ma
pip install -r requirements
```

### Installation of external repositories
haystack

```

git clone https://github.com/deepset-ai/haystack.git
cd haystack
pip install --upgrade pip
pip install -e '.[all-gpu]'
```

trec evaluation

```

git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
```

ranking utils
```

git clone https://github.com/mrjleo/ranking-utils.git
cd ranking-utils
python -m pip install .
```
Run minimal example
```
CUDA_VISIBLE_DEVICES=x,y python src/minimal_example.py
```
