import numpy as np
import matplotlib.pyplot as plt
TINY = 1e-12

def normalize(X, mean=None, stddev=None, useful_features=None, remove_constant=True):
    calc_mean, calc_stddev = False, False
    
    if mean is None:
        mean = np.mean(X, 0) # training set
        calc_mean = True
    
    if stddev is None:
        stddev = np.std(X, 0) # training set
        calc_stddev = True
        useful_features = np.nonzero(stddev)[0] # inconstant features, ([0]=shape correction)
    
    if remove_constant and useful_features is not None:
        X = X[:, useful_features]
        if calc_mean:
            mean = mean[useful_features]
        if calc_stddev:
            stddev = stddev[useful_features]
    
    X_zm = X - mean    
    X_zm_unit = X_zm / stddev
    
    return X_zm_unit, mean, stddev, useful_features

def norm_entropy(p):
    '''p: probabilities '''
    n = p.shape[0]
    return - p.dot(np.log(p + TINY) / np.log(n + TINY))

def entropic_scores(r):
    '''r: relative importances '''
    r = np.abs(r)
    ps = r / np.sum(r, axis=0) # 'probabilities'
    hs = [1-norm_entropy(p) for p in ps.T]
    return hs

def mse(predicted, target):
    ''' mean square error '''
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted #(n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target #(n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0] #value not array

def rmse(predicted, target):
    ''' root mean square error '''
    return np.sqrt(mse(predicted, target))

def nmse(predicted, target):
    ''' normalized mean square error '''
    return mse(predicted, target) / np.var(target)

def nrmse(predicted, target):
    ''' normalized root mean square error '''
    return rmse(predicted, target) / np.std(target)

def print_table_pretty(name, values, factor_label, model_names):
    headers = [factor_label + str(i) for i in range(len(values[0]))]
    headers[-1] = "Avg."
    headers = "\t" + "\t".join(headers)
    print("{0}:\n{1}".format(name, headers))
    
    for i, values in enumerate(values):
        value = ""
        for v in values:
            value +=  "{0:.2f}".format(v) + "&\t"
        print("{0}\t{1}".format(model_names[i], value))
    print("") #newline

def subset_of_data(X, y, n_samples, rng=None):
    if rng is None:
        rng = np.random.RandomState(123)
    i = range(len(y))
    rng.shuffle(i)
    i = i[:n_samples]
    return X[i], y[i]
        
def surface_plot(X, Y, Z, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.show()
        
def get_factor_name(factor):
    if factor == 0:
        return 'Azimuth'
    elif factor == 1:
        return 'Elevation'
    elif factor == 2:
        return 'Red'
    elif factor == 3:
        return 'Green'
    elif factor == 4:
        return 'Blue'
    else:
        raise Exception("Invalid factor, please choose integer in range (0, 4)")

def get_angle(sin_a, cos_a):
    return np.arctan2(sin_a, cos_a)
    
def angle_error(predicted, target, var=1.):
    PredAz = get_angle(predicted[:, 0], predicted[:, 1]) 
    diff = target - PredAz
    a_error = (diff + np.pi) % (2*np.pi) - np.pi
    #a_error = np.arctan2(np.sin(target - PredAz), np.cos(target - PredAz))   
    return np.sqrt(np.mean(a_error**2) / var)

def save_plot(plt, model, input_type, factor, name="hinton", path="plots/hinton/"):
    filename = os.path.join(path, "{0}_{1}_{2}_{3}.pdf".format(name, model, factor, input_type))
    plt.savefig(filename)
    
def save_plot(plt, model, name="hinton", path="plots/hinton/"):
    filename = os.path.join(path, "{0}_{1}.pdf".format(name, model))
    plt.savefig(filename)
    
def save_weights(weights, model_name, input_type, factor, path="weights/"):
    filename = os.path.join(path, "w_{0}_{1}_{2}".format(model_name, factor, input_type))
    np.savez(filename, weights=weights)
    
def load_weights(model_name, input_type, factor, path="weights/"):
    filename = os.path.join(path, "w_{0}_{1}_{2}.npz".format(model_name, factor, input_type))
    return np.load(filename)['weights']