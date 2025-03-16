import wandb
from train import main 
import argparse

parser = argparse.ArgumentParser(description='Accept Command line arguments of wandb_project, wandb_entity, dataset')
parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')    
parser.add_argument('-we','--wandb_entity', type=str, default='myname',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')  
parser.add_argument('-d','--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help="Choose dataset from ['mnist', 'fashion_mnist']")

args = parser.parse_args()

sweep_config = {
    'method': 'bayes',
}

# Define the metric configuration
metric = {
    'name': 'best_val_accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = metric
    
# Define the hyperparameters (parameters) dictionary
parameters_dict = {
    'dataset': {'values': [f'{args.dataset}']},
    'num_layers': {'values': [3, 4, 5]},
    'hidden_size': {'values': [32, 64, 128]},
    'learning_rate': {'values': [1e-3, 1e-4]},
    'weight_init': {'values': ["random", "Xavier"]},
    'weight_decay': {'values': [0, 0.0005, 0.5]},
    'epochs': {'values': [5, 10]},
    'batch_size': {'values': [16, 32, 64]},
    'activation': {'values': ["sigmoid", "tanh", "ReLU"]},
    'loss':{'values': ['cross_entropy', 'mean_squared_error']},
    'optimizer': {'values': ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
    'momentum': {'values': [0.5, 0.9, .999]},
    'beta': {'values': [0.5, 0.9, .999]},
    'beta1': {'values': [0.5, 0.9, .999]},
    'beta2': {'values': [0.5, 0.9, .999]}, 
    'epsilon': {'values': [0.000001, 0.0000001]},
}


sweep_config['parameters'] = parameters_dict
sweep_id= wandb.sweep(sweep_config, entity=args.wandb_entity, project=args.wandb_project)
print(sweep_id)
wandb.agent(sweep_id, function=main)