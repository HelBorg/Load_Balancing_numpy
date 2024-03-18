import collections
import os
import pickle


def save_pickle(var, path):
    current_path = os.path.realpath(__file__)
    project_dir = "\\".join(current_path.split("\\")[:-2])
    with open(project_dir + path, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def upload_pickle(file_path):
    current_path = os.path.realpath(__file__)
    project_dir = "\\".join(current_path.split("\\")[:-2])
    with open(project_dir + file_path, 'rb') as handle:
        var = pickle.load(handle)
    return var


def safe_list_get(l, idx: int, default):
    try:
        return l[idx]
    except IndexError:
        return default
