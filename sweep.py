import wandb
from train import main 

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
    'project'
    'epochs': {'values': [5, 10]},
    'num_layers': {'values': [3, 4, 5]},
    'hidden_size': {'values': [32, 64, 128]},
    'weight_decay': {'values': [0, 0.0005, 0.5]},
    'learning_rate': {'values': [1e-3, 1e-4]},
    'optimizer': {'values': ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
    'batch_size': {'values': [16, 32, 64]},
    'weight_init': {'values': ["random", "Xavier"]},
    'activation': {'values': ["sigmoid", "tanh", "ReLU"]},
    'loss':{'values': ['cross_entropy', 'mean_squared_error']},
    'momentum': {'values': [0.5, 0.9, .999]},
    'beta': {'values': [0.5, 0.9, .999]},
    'beta1': {'values': [0.5, 0.9, .999]},
    'beta2': {'values': [0.5, 0.9, .999]},
    'epsilon': {'values': [0.000001, 0.0000001]},
}

sweep_config['parameters'] = parameters_dict
sweep_id= wandb.sweep(sweep_config, project="Programming_Assignment_1")
print(sweep_id)
wandb.agent(sweep_id, function=main)