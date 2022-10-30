def FedAvg(model_state_dicts):
    '''
        Calculates and generates the FedAvg of the state_dict of a list of models. Returns the FedAvg state_dict.
    '''
    # Sum up tensors from all states
    state_dict_sum = {}  # Stores the sum of state parameters
    for state_dict in model_state_dicts:
        for key, params in state_dict.items():
            if key in state_dict_sum:
                state_dict_sum[key] += params.detach().clone()
            else:
                state_dict_sum[key] = params.detach().clone()

    # Get Federated Average of clients' parameters
    state_dict_avg = {}
    for key in state_dict_sum:
        state_dict_avg[key] = state_dict_sum[key] / len(model_state_dicts)

    return state_dict_avg