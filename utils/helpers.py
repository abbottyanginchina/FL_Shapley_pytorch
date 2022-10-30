from itertools import chain, combinations, permutations

def powerset(iterable, no_null = True):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1 if no_null else 0, len(s)+1))

def grangerset(iterable):
    s = iterable
    s = combinations(s, len(s)-1)
    return s

def getAllClients(iterable):
    s = tuple(iterable)
    return s

def aggListOfDicts(lst):
    '''
        Combines a list of dictionaries into a dictionary of lists
    '''
    agg = {}
    for dct in lst:
        for name, val in dct.items():
            if name in agg:
                agg[name] += [val]
            else:
                agg[name] = [val]

    return agg