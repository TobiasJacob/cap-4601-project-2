import json
import logging

import numpy as np

logger = logging.getLogger("root")


def batchify(samples, batch_size):
    """
    Batchfy samples with a batch size
    """
    num_samples = len(samples)

    list_samples_batches = []

    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]["tokens"]) > 350:
            to_single_batch.append(i)

    for i in to_single_batch:
        logger.info(
            "Single batch sample: %s-%d",
            samples[i]["doc_key"],
            samples[i]["sentence_ix"],
        )
        list_samples_batches.append([samples[i]])
        samples.remove(samples[i])

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i : i + batch_size])

    assert sum([len(batch) for batch in list_samples_batches]) == num_samples

    return list_samples_batches


def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_train_fold(data, fold):
    print("Getting train fold %d..." % fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold + 1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < l or i >= r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print("# documents: %d --> %d" % (len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data


def get_test_fold(data, fold):
    print("Getting test fold %d..." % fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold + 1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= l and i < r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print("# documents: %d --> %d" % (len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data
