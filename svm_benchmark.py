import cPickle
import numpy as np

    
def make_cv_dataset(datasets, cv, batch_size=200):
    train_set = []
    test_set = []

    for doc in datasets:
        if doc["cv"] == cv:
            test_set.append(doc)
        else:
            train_set.append(doc)

    # Divid 10% of the training set as valid
    np
    n_batches = len(train_set)
    if n_batches % batch_size > 0:
        extra_num = batch_size - n_batches % batch_size
        data = np.random.permutation(train_set)
        extra_data = data[:extra_num]
        new_data = data + extra_data
    else:
        new_data = train_set

    new_data = np.random.permutation(new_data)
    n_batches = len(new_data)/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    train_set = new_data[:n_train_batches*batch_size]
    valid_set = new_data[n_train_batches*batch_size:]
    return train_set, valid_set, test_set

def load_data(dataset_file):
    with open(dataset_file) as df:
        docs = cPickle.load(df)
        vocab = cPickle.dump(vocab, df)
        type2id = cPickle.dump(type2id, df)
        pop2id = cPickle.dump(pop2id, df)
        datasets = [docs, vocab, type2id, pop2id]
    return datasets