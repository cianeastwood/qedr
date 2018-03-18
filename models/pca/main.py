import numpy as np
import os, sys
from operator import mul
from sklearn.decomposition import PCA
sys.path.append("..")
sys.path.append("../..")
from lib.eval.regression import normalize

if __name__ == '__main__':
    # Sys args: [filename, n_components, gaps, train_fract, data_dir]

    try:
        n_components = int(sys.argv[1])
    except Exception:
        n_components = 6
    print("Using {0} components.".format(n_components))

    try:
        gaps = True if str(sys.argv[2]).strip() == "True" else False
    except Exception:
        gaps = True
    print("Using train data with gaps: {0}".format(gaps))

    try:
        train_fract = float(sys.argv[3])
    except Exception:
        train_fract = 0.8
    print("Using {0} as train_fract.".format(train_fract))

    try:
        data_dir = str(sys.argv[4])
    except Exception:
        data_dir = '../../data/'
    print("Loading data from {0}.".format(data_dir))

    # Load data
    def load_data():
        data = np.load(os.path.join(data_dir, 'teapots.npz'))['images']
        if len(data.shape) > 2:
            shape = reduce(mul, data.shape[1:], 1)
        data = data.reshape([-1, shape])
        return data

    data = load_data()

    # Get training data
    n_train = int(len(data) * train_fract)
    data = data[:n_train] #reuse var to save memory
    if gaps:
        gap_ids = np.load(os.path.join(data_dir, 'gap_ids.npy'))
        data = np.delete(data, gap_ids, 0)

    # Fit model to normalized training data
    data, m, s, fs = normalize(data)
    rng = np.random.RandomState(123)
    model = PCA(n_components=n_components, random_state=rng)
    model.fit(data)

    # Encode all (normalized) data
    data = load_data()
    data, _, _, _   = normalize(data, m, s, fs)
    data = model.transform(data) # codes, reuse var
    expl_var = np.array(np.sum(model.explained_variance_ratio_))

    # Save results
    codes_dir = os.path.join(data_dir, 'codes/')
    f_name = os.path.join(codes_dir, "pca_{0}_{1}v".format(gaps, n_components))
    np.savez_compressed(f_name, codes=data, expl_var=expl_var)
    print("PCA values saved to {1}.".format(f_name))
