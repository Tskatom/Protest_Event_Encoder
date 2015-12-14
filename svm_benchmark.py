import cPickle
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from collections import namedtuple
from gensim.models import Doc2Vec


def make_cv_dataset(datasets, cv, batch_size=200):
    train_set = []
    test_set = []

    for doc in datasets:
        if doc["cv"] == cv:
            test_set.append(doc)
        else:
            train_set.append(doc)

    # Divid 10% of the training set as valid
    np.random.seed(1234)
    n_batches = len(train_set)
    if n_batches % batch_size > 0:
        extra_num = batch_size - n_batches % batch_size
        data = np.random.permutation(train_set).tolist()
        extra_data = data[:extra_num]
        new_data = data + extra_data
    else:
        new_data = train_set

    new_data = np.random.permutation(new_data).tolist()
    n_batches = len(new_data)/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    train_set = new_data[:n_train_batches*batch_size]
    valid_set = new_data[n_train_batches*batch_size:]
    return train_set, valid_set, test_set

def load_data(dataset_file):
    with open(dataset_file) as df:
        docs = cPickle.load(df)
        vocab = cPickle.load(df)
        type2id = cPickle.load(df)
        pop2id = cPickle.load(df)
        datasets = [docs, vocab, type2id, pop2id]
    return datasets

def svm_experiment(train_set, valid_set, test_set):
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set
    
    c_range = np.logspace(-2, 1, 4)
    # linear svms
    best_score = 0.0
    best_model = None
    for c in c_range:
        model = svm.LinearSVC(C=c)
        model.fit(train_set_x, train_set_y)
        valid_pred = model.predict(valid_set_x)
        score = 1 - np.mean(np.not_equal(valid_pred, valid_set_y))
        if score > best_score:
            best_score = score
            best_param = {"kernel": 'linear', "C": c}
            best_model = model

    """
    # rbf kernel
    gamma_range = np.logspace(-9, 2, 5)
    for c in c_range:
        for gamma in gamma_range:
            print gamma
            model = svm.SVC(C=c, gamma=gamma, kernel='rbf')
            model.fit(train_set_x, train_set_y)
            valid_pred = model.predict(valid_set_x)
            score = 1 - np.mean(np.not_equal(valid_pred, valid_set_y))
            if score > best_score:
                best_score = score
                best_param = {"kernel": 'rbf', "C": c, "gamma": gamma}
                best_model = model
    """
    test_pred = best_model.predict(test_set_x)
    test_score = 1 - np.mean(np.not_equal(test_set_y, test_pred))
    print "Test Performance %f under Best valid Score: %f" % (test_score, best_score), " Best Params: ", best_param
    return test_score, best_param


def svm_doc2vec(dataset_file, train_epochs=100, cores=4):
    datasets = load_data(dataset_file)
    docs = datasets[0][:10000]
    event = namedtuple('eventDoc', 'words tags')
    event_docs = [event(d['tokens'], [i]) for i , d in enumerate(docs)]
    doc2vec_model = Doc2Vec(alpha=0.025, min_alpha=0.025, workers=cores)
    # train the language model 100 epochs
    doc2vec_model.build_vocab(event_docs)
    for epoch in range(train_epochs):
        print epoch
        doc2vec_model.train(event_docs)
        doc_vecs = [doc2vec_model.docvecs[d.tags[0]] for d in event_docs]

        new_docs = []
        for idx, doc in enumerate(docs):
            doc['vec'] = doc_vecs[idx]
            new_docs.append(doc)

        train_set, valid_set, test_set = make_cv_dataset(new_docs, 0)
        print len(train_set), len(valid_set), len(test_set)
        train_set_x = np.asarray([d['vec'] for d in train_set])
        train_set_pop = [d['pop'] for d in train_set]
        train_set_type = [d['etype'] for d in train_set]
        valid_set_x = np.asarray([d['vec'] for d in valid_set])
        valid_set_pop = [d['pop'] for d in valid_set]
        valid_set_type = [d['etype'] for d in valid_set]
        test_set_x = np.asarray([d['vec'] for d in test_set])
        test_set_pop = [d['pop'] for d in test_set]
        test_set_type = [d['etype'] for d in test_set]

        print "Start compute the Population Performance"
        svm_experiment([train_set_x, train_set_pop], [valid_set_x, valid_set_pop], [test_set_x, test_set_pop])

        print "Start compute the Event Type Performance"
        svm_experiment([train_set_x, train_set_type], [valid_set_x, valid_set_type], [test_set_x, test_set_type])
    
    # save the model
    doc2vec_model.save('./data/doc2vec_%d' % epoch)

if __name__ == "__main__":
    dataset_file = "./data/svm_dataset"
    #svm_doc2vec(dataset_file, )

