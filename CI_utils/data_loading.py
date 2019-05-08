
def dict_as_pickle(dict, filename):
    '''
    :param dict: dictionary to be saved
    :param filename: Requires to be .pkl
    '''
    assert('.pkl' in filename)

    import pickle
    with open(filename, 'wb') as pickle_file:
        pickle.dump(dict, pickle_file)
        pickle_file.close()


def from_pickle(filename):
    '''
    :param filename: Requires to be .pkl
    :return: loaded object
    '''
    assert ('.pkl' in filename)

    import pickle
    with open(filename, 'rb') as pickle_file:
        obj = pickle.load(pickle_file)
        pickle_file.close()

    return obj
