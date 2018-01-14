import numpy as np
import os, sys
from model import encode
from lib.eval.regression import normalize
      
if __name__ == '__main__':
    # Sys args: [filename, m_name, n_components, gaps, train_fract, data_dir]
    try:
        m_name = sys.argv[1]
    except Exception:
        m_name = 'pca'
    print("Using {0} model.".format(m_name.upper()))
    
    try:
        n_components = int(sys.argv[2])
    except Exception:
        n_components = 6
    print("Using {0} components.".format(n_components))
        
    try:
        gaps = True if str(sys.argv[3]).strip() == "True" else False
    except Exception:
        gaps = False
    print("Using train data with gaps: {0}".format(gaps))
        
    try:
        train_fract = float(sys.argv[4])
    except Exception:
        train_fract = 0.8
    print("Using {0} as train_fract.".format(train_fract))
    
    try:
        data_dir = str(sys.argv[5])
    except Exception:
        data_dir = '../../data'
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
    results_dict = encode(m_name, n_components, train, data)
    
    # Save results
    codes_dir = os.path.join(data_dir, 'codes/')
    filename = os.path.join(codes_dir, "{0}_{1}_{2}_{3}v_test".format(m_name, gaps, n_components))
    np.savez_compressed(filename, **results_dict)
    print("{0} values saved to {1}.".format(m_name, filename))
