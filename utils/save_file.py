import os, glob
import torch

# Initialize system and define helper functions
def createDirectory(path):
    pathExists = os.path.exists(path)
    if not pathExists:
        print(f'"{path}" does not exist.')
        os.makedirs(path)
        print(f'"{path}" created.')

# Delete existing .pt files from previous run
def deleteAllModels(path):
    filepaths = glob.glob(f'{path}/**/*.pt', recursive=True)
    for filepath in filepaths:
        os.remove(filepath)
        print(f'"{filepath}" deleted.')

# Define checkpoint functions (Simulates data exchange between clients and server)
def saveCheckpoint(name, model_state_dict, optimizer_state_dict, filepath, verbose=False):
    '''
        Saves state dictionaries of model and optimizer as a .pt file
    '''
    torch.save({
        'name': name,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
    }, filepath)

    if verbose:
        print(f'\n"{name}" model saved as "{filepath}".\n')

    return True


def loadCheckpoint(filepath, verbose=False):
    '''
        Loads and returns the state dictionaries of model and optimizer from a .pt file
    '''
    checkpoint = torch.load(filepath)

    if verbose:
        name = checkpoint['name']
        print(f'\n"{name}" model loaded from "{filepath}".\n')

    return checkpoint


def print_parameters(model):
    '''
        Outputs the learnable parameter counts for each layer and in total
    '''
    print('Model Layer Parameters:\n')
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        print(f'{name} - {params} parameters')
        total_params += params
    print(f'\n>>Total - {total_params} parameters\n')