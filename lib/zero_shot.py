import numpy as np
import os

def elev_gap(gts):
    return gts[1] > np.pi / 4.

def elev_gap_large(gts):
    return gts[1] > np.pi / 2.5

def colour_gap(gts):
    return gts[2] > gts[3] + 0.15 and gts[2] > gts[4] + 0.15

def colour_gap_large(gts):
    return gts[2] > gts[3] + 0.7 and gts[2] > gts[4] + 0.7

def get_gap_ids(all_gts):
    gap_ids = []
    for i, gts in enumerate(all_gts):
        if elev_gap(gts) and colour_gap(gts):
            gap_ids.append(i)
    return gap_ids

def get_large_gap_ids(all_gts):
    ''' Extreme examples. '''
    gap_ids = []
    for i, gts in enumerate(all_gts):
        if elev_gap_large(gts) and colour_gap_large(gts):
            gap_ids.append(i)
    return gap_ids

def get_code_space_gap_ids(codes, n_samples=64):
    ''' Get extreme codes in a similar code space gap.'''
    r_g_diff = codes[:, 2] - codes[:, 3]
    r_b_diff = codes[:, 2] - codes[:, 4]
    r_diff = r_g_diff + r_b_diff - (codes[:, 1] / 2.)
    s_i = np.argsort(-r_diff)
    return codes[s_i][:n_samples]

if __name__ == '__main__':
    data_dir = os.path.join(os.environ['PYTHONPATH'], 'data')
    try:
        data_dir = str(sys.argv[4])
    except Exception:
        print("Warning: loading data from {0} as no path was passed.".format(data_dir))

    # load gts
    gts = np.load(os.path.join(data_dir, 'teapots.npz'))['gts']
    gap_ids = get_gap_ids(gts)
    large_gap_ids = get_large_gap_ids(gts)
    
    # save gap_ids
    np.save(os.path.join(data_dir, 'gap_ids'), gap_ids)
    np.save(os.path.join(data_dir, 'large_gap_ids'), large_gap_ids)
    
    # print stats
    print("Images in gap: {0}/{1}".format(len(gap_ids), len(gts)))
    print("Images in large gap: {0}/{1}".format(len(large_gap_ids), len(gts)))