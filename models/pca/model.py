import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA as ICA

def encode(f_name, n_components, train, all_data, rng=None):
    rng = np.random.RandomState(123) if rng is None else rng
    
    if f_name == 'pca':
            f = PCA
    elif f_name == 'ica':
            f = ICA
            
    model = f(n_components=n_components, random_state=rng)
    model.fit(train)
    codes = model.transform(all_data)
    
    values = [codes]
    keys = ['codes']
    if f_name == 'pca':
        values.append(np.array(np.sum(model.explained_variance_ratio_)))
        keys.append('explained_variance')
    
    return dict(zip(keys, values))