import torch
import sys,os
from utils.save_file import createDirectory, saveCheckpoint

class server():
    def __init__(self, FederatedModel, FederatedLossFunc, FederatedOptimizer, FederatedLearnRate, FederatedMomentum, FederatedWeightDecay):
        self.FederatedModel = FederatedModel
        self.FederatedLossFunc = FederatedLossFunc
        self.FederatedOptimizer = FederatedOptimizer
        self.FederatedLearnRate = FederatedLearnRate
        self.FederatedMomentum = FederatedMomentum
        self.FederatedWeightDecay = FederatedWeightDecay


    def initServer(self, model_path, folder_name, dataloader):
        '''
            Initializes server model and returns object with attributes
        '''
        print('Initializing server model...')

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Spawn server model and functions
        server_name = 'server'
        server_model = self.FederatedModel().to(device)
        server_loss_func = self.FederatedLossFunc()
        server_optimizer = self.FederatedOptimizer(server_model.parameters(), lr=self.FederatedLearnRate,
                                              momentum=self.FederatedMomentum, weight_decay=self.FederatedWeightDecay)
        server_dataloader = dataloader

        print(server_model, '\n')
        print(server_optimizer)

        createDirectory(f'{model_path}/{folder_name}/server')
        createDirectory(f'{model_path}/{folder_name}/client')

        # Collect objects into a reference dictionary
        server = {
            'name': server_name,
            'model': server_model,
            'dataloader': server_dataloader,
            'optimizer': server_optimizer,
            'loss_func': server_loss_func,
            'filepath': f'{model_path}/{folder_name}/server/server_model.pt',
            'client_filepath': f'{model_path}/{folder_name}/client'
        }

        # Save server model state_dicts (simulating public access to server model parameters)
        saveCheckpoint(
            server_name,
            server_model.state_dict(),
            server_optimizer.state_dict(),
            server['filepath'],
            verbose=True
        )

        return server
