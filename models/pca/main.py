import numpy as np
import os, sys
from sklearn.decomposition import PCA
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
        data_dir = '/disk/scratch1/cian/data/'#'../../data'
    print("Loading data from {0}.".format(data_dir))
        
    # Load data
    data = np.load(os.path.join(data_dir, 'teapots.npz'))['images']
    
    # Split data
    n_train = int(len(data) * train_fract)
    train = data[:n_train]        
    if gaps:
        gap_ids = np.load(os.path.join(data_dir, 'gap_ids.npy'))
        train = np.delete(train, gap_ids, 0)
    
    # Normalize data
    train, m, s, _ = normalize(train)
    data, _, _, _   = normalize(data, m, s)
    
    # Encode
    rng = np.random.RandomState(123)
    model = PCA(n_components=n_components, random_state=rng)
    model.fit(train)
    data = model.transform(data) # codes, reuse var
    expl_var = np.array(np.sum(model.explained_variance_ratio_))

    # Save results   
    codes_dir = os.path.join(data_dir, 'codes/')
    filename = os.path.join(codes_dir, "pca_{0}_{1}v_new_T".format(gaps, n_components))
    np.savez_compressed(filename, codes=data, expl_var=expl_var)
    print("{0} values saved to {1}.".format(m_name, filename))
