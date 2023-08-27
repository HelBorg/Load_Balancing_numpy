import pickle


def save_pickle(var, path):
    with open(path, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def upload_pickle(file_path):
    with open(file_path, 'rb') as handle:
        var = pickle.load(handle)
    return var


def safe_list_get(l, idx: int, default):
    try:
        return l[idx]
    except IndexError:
        return default
