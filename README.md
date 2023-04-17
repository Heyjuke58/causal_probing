# Causal Probing

## Setup

Setup Conda environment

```

conda create -n ma pyton=3.8
conda activate ma
pip install -r requirements
python -m spacy download en_core_wb_sm
```

### Installation of external repositories

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

neuralcoref

```sh
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
```

Run minimal example

```
CUDA_VISIBLE_DEVICES=x,y python src/minimal_example.py
```
