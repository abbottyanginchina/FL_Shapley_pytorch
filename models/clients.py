def initClients(num_norm, num_free, num_avsl, server, dataloaders):
    '''
        Initializes clients objects and returns a list of client object
    '''

    print('Initializing clients...')
    # Setup client devices
    behaviour_list = [
        *['NORMAL' for i in range(num_norm)],
        *['FREERIDER' for i in range(num_free)],
        *['ADVERSARIAL' for i in range(num_avsl)],
    ]

    clients = []
    for n, behaviour in enumerate(behaviour_list):
        # Spawn client model and functions
        client_name = f'client_{n}'

        # Collect client's objects into a reference dictionary
        clients += [{
            'name': client_name,
            'behaviour': behaviour,
            'filepath': f'{server["client_filepath"]}/{client_name}.pt',
            'dataloader': dataloaders[n]
        }]

    print('Client Name / Behaviour:', [(client['name'], client['behaviour']) for client in clients], '\n')

    return clients

