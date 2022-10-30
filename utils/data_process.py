# Functions to split dataset based on IID or Non-IID selection
import random

def prepareIID(dataset, num_clients):
    '''
        Prepares IID training datasets for each client
    '''
    dataset_split = [[] for i in range(num_clients)]

    for idx, sample in enumerate(dataset):
        dataset_split[idx % num_clients] += [sample]

    return dataset_split

def prepareNIID1(dataset, num_clients):
    '''
        Prepares NIID-1 training datasets for each client (Overlapping sample sets)
    '''
    dataset_split = [[] for i in range(num_clients)]

    for idx, sample in enumerate(dataset):
        dataset_split[idx % num_clients] += [random.choice(dataset)]

    return dataset_split


def prepareNIID2(dataset, num_clients):
    '''
        Prepares NIID-1 training datasets for each client (Unequal data distribution)
    '''
    dataset_split = [[] for i in range(num_clients)]

    for idx, sample in enumerate(dataset):
        dataset_split[random.randint(0, num_clients - 1)] += [sample]

    return dataset_split


def prepareNIID12(dataset, num_clients):
    '''
        Prepares NIID-1+2 training datasets for each client
        (Overlapping sample sets + Unequal data distribution)
    '''
    dataset_split = [[] for i in range(num_clients)]

    for sample in dataset:
        dataset_split[random.randint(0, num_clients - 1)] += [random.choice(dataset)]

    return dataset_split