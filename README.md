## Quantitative Evaluation of Disentangled Representations

Code to reproduce the results in our ICLR 2018 paper: [A Framework for the Quantitative Evaluation of Disentangled Representations](https://openreview.net/forum?id=By-7dz-AZ).

## Prerequisites

- Python 2.7.5+/3.5+, NumPy, TensorFlow 1.0+, SciPy, Matplotlib, Scikit-learn

## Data

- Download [here](https://www.dropbox.com/s/woeyomxuylqu7tx/edinburgh_teapots.zip?dl=0).
  - If RAM < 10GB, convert .npz to .jpeg before training to load batches of images into memory (rather than entire dataset)
    - `python npz_to_jpeg.py` (after editing paths)
- Generated using [this](https://github.com/polmorenoc/inversegraphics) renderer.

## Models

- PCA
- [InfoGAN](https://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf)
- [VAE](https://arxiv.org/pdf/1312.6114.pdf) / [\beta-VAE](https://openreview.net/pdf?id=Sy2fzU9gl)

#### Train

- `PYTHONPATH=[/path/to/qedr/] python main.py`

#### Save codes

- `PYTHONPATH=[/path/to/qedr/] python main.py --save_codes`

## Quantitative Evaluation

- quantify.ipynb
