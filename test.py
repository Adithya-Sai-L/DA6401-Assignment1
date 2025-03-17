import numpy as np

import argparse
from model import NeuralNetwork
from optimizers import create_optimizer
from keras.datasets import fashion_mnist, mnist
import wandb
import plotly.graph_objects as go


def create_parser():
    parser = argparse.ArgumentParser(description='Accept Command line arguments as mentioned in Code Specifications')
    parser.add_argument('-we','--wandb_entity', type=str, default='myname',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',                    
                        help='Project name used to track experiments in Weights & Biases dashboard')
    # Added arguments as per Code Specifications of Assignment
    parser.add_argument('-d','--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help="Choose dataset from ['mnist', 'fashion_mnist']")
    # parser.add_argument('-e','--epochs', type=int, default=1, help="Number of epochs to train neural network.")
    # parser.add_argument('-b','--batch_size', type=int, default=4, help='Batch size used to train neural network.')
    parser.add_argument('-l','--loss', type=str, default='cross_entropy', choices=["mean_squared_error", "cross_entropy"], help="Choose loss from ['mean_squarred_error', 'cross_entropy']")                        
    parser.add_argument('-nhl','--num_layers', type=int, default=1, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument('-sz','--hidden_size', type=int, default=4, help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument('-a','--activation', type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "ReLU"], help='Choose activation from ["identity", "sigmoid", "tanh", "ReLU"]')    
    parser.add_argument('-w_i','--weight_init', type=str, default="random", choices=["random", "Xavier"], help="Choose initialization from ['random', 'Xavier']")
    parser.add_argument('-b','--batch_size', type=int, default=4, help='Batch size used to train neural network.')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.1, help="Learning rate used to optimize model parameters")
    parser.add_argument('-o','--optimizer', type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help='Choose optimizer from ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
    parser.add_argument('-w_d','--weight_decay', type=float, default=.0, help="Weight decay used by optimizers.")
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to the weights file')
    return parser


def one_hot_encoding(labels, num_classes=10):
    Y = np.zeros((len(labels), num_classes))
    for i, j in enumerate(labels):
        Y[i,j] = 1
    return Y


def main():
    parser = create_parser()
    args = parser.parse_args()
    



    # Check the parsed arguments to be used in training script
    layer_sizes = [784] + [args.hidden_size]*args.num_layers + [10]
    model = NeuralNetwork(layer_sizes, args)
    
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() if args.dataset == "fashion_mnist" else mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] if args.dataset == "fashion_mnist" else ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    params = np.load(args.weights, allow_pickle=True)
    
    model.weights = params.item().get('weights')
    model.biases = params.item().get('biases')

    X_test = (test_images.astype(np.float32)-127)/128.
    # Y_test = one_hot_encoding(test_labels)

    confusion_matrix = np.zeros((10,10))
    for (image, label) in zip(X_test, test_labels):
        y_hat = model.forward(image.flatten())[-1]
        confusion_matrix[np.argmax(y_hat)][label] += 1

    labels = class_names

    num_classes = confusion_matrix.shape[0]
    
    # Calculate per-class metrics (Recall, Precision, F1)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    recalls = np.zeros(num_classes)
    precisions = np.zeros(num_classes)
    f1_scores = np.zeros(num_classes)
    for i in range(num_classes):
        recalls[i] = confusion_matrix[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0
        col_sum = confusion_matrix[:, i].sum()
        precisions[i] = confusion_matrix[i, i] / col_sum if col_sum > 0 else 0.0
        if recalls[i] + precisions[i] > 0:
            f1_scores[i] = 2 * recalls[i] * precisions[i] / (recalls[i] + precisions[i])
        else:
            f1_scores[i] = 0.0

    # Create separate matrices for correct predictions (diagonals) and misclassifications (off-diagonals)
    diag_matrix = np.full(confusion_matrix.shape, np.nan)
    mis_matrix = confusion_matrix.copy()
    for i in range(num_classes):
        diag_matrix[i, i] = confusion_matrix[i, i]
        mis_matrix[i, i] = np.nan

    # Build customdata for the diagonal cells (for hover display of metrics)
    customdata = np.empty((num_classes, num_classes, 3))
    customdata[:] = np.nan
    for i in range(num_classes):
        customdata[i, i, 0] = recalls[i]
        customdata[i, i, 1] = precisions[i]
        customdata[i, i, 2] = f1_scores[i]

    # Heatmap for correct predictions with detailed hover info
    trace_diag = go.Heatmap(
        z=diag_matrix.tolist(),
        x=labels,
        y=labels,
        customdata=customdata.tolist(),
        hoverongaps=False,
        hovertemplate=(
            "Predicted %{z} %{y}s just right<br>" +
            "Recall: %{customdata[0]:.2f}<br>" +
            "Precision: %{customdata[1]:.2f}<br>" +
            "F1: %{customdata[2]:.2f}<extra></extra>"
        ),
        coloraxis="coloraxis2",  # Green color axis for correct predictions
        showscale=True,
        name="Correct Predictions"
    )
    
    # Heatmap for misclassifications with custom hover text
    trace_mis = go.Heatmap(
        z=mis_matrix.tolist(),
        x=labels,
        y=labels,
        hoverongaps=False,
        hovertemplate="Predicted %{z} %{x}s as %{y}s<br><extra></extra>",
        coloraxis="coloraxis",  # Red color axis for misclassifications
        showscale=True,
        name="Incorrect"
    )
    
    # Create base figure with both traces
    fig = go.Figure(data=[trace_diag, trace_mis])
    
    # Update layout (axes, colorbars, and background)
    fig.update_layout(
        title={"text": "Confusion Matrix", "x": 0.5},
        width=750,
        height=600,
        xaxis=dict(
            title="True Label",
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            showgrid=False
        ),
        yaxis=dict(
            title="Predicted Label",
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            showgrid=False
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Misclassification color axis (red)
        coloraxis=dict(
            colorscale=[[0, "rgba(180, 0, 0, 0.05)"], [1, "rgba(180, 0, 0, 0.58)"]],
            colorbar=dict(title="Incorrect", x=1.00)
        ),
        # Correct predictions color axis (green)
        coloraxis2=dict(
            colorscale=[[0, "rgba(0, 180, 0, 0.44)"], [1, "rgba(0, 180, 0, 1)"]],
            colorbar=dict(title="Correct Predictions", x=1.12)
        )
    )
    
    # "View Mode" dropdown: Toggle full matrix, only correct, or only misclassifications.
    view_mode_buttons = [
        dict(
            label="Show All",
            method="update",
            args=[{"visible": [True, True]}, 
                  {"coloraxis.colorbar.visible": True,
                   "coloraxis2.colorbar.visible": True}]
        ),
        dict(
            label="Only Correct",
            method="update",
            args=[{"visible": [True, False]},
                  {"coloraxis.colorbar.visible": False,
                   "coloraxis2.colorbar.visible": True}]
        ),
        dict(
            label="Only Misclassifications",
            method="update",
            args=[{"visible": [False, True]},
                  {"coloraxis.colorbar.visible": True,
                   "coloraxis2.colorbar.visible": False}]
        )
    ]
    
    # "Class Focus" dropdown: Select a class to focus on or "All Classes" for full view.
    focus_buttons = []
    focus_buttons.append(dict(
        label="All Classes",
        method="update",
        args=[{"z": [diag_matrix.tolist(), mis_matrix.tolist()]},
              {"annotations": []}]
    ))
    
    # For each class, mask the matrix so only the row or column for that class remains.
    for idx, lab in enumerate(labels):
        mask = np.zeros(confusion_matrix.shape, dtype=bool)
        indices = np.indices(confusion_matrix.shape)
        mask[(indices[0] == idx) | (indices[1] == idx)] = True
        
        diag_focus = np.where(mask, diag_matrix, np.nan)
        mis_focus = np.where(mask, mis_matrix, np.nan)
        
        # Annotation for selected class (showing class name and metrics)
        ann_text = (
            f"<b>{lab}</b><br>"
            f"Recall: {recalls[idx]*100:.1f}%<br>"
            f"Precision: {precisions[idx]*100:.1f}%<br>"
            f"F1: {f1_scores[idx]*100:.1f}%"
        )
        annotation = dict(
            xref="paper", yref="paper",
            x=0.02, y=0.95,
            text=ann_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)"
        )
        
        focus_buttons.append(dict(
            label=lab,
            method="update",
            args=[{"z": [diag_focus.tolist(), mis_focus.tolist()]},
                  {"annotations": [annotation],
                   "coloraxis.colorbar.visible": False,
                   "coloraxis2.colorbar.visible": False}]
        ))
    
    # Add dropdown menus with opaque backgrounds.
    fig.update_layout(
        updatemenus=[
            {
                "buttons": view_mode_buttons,
                "direction": "down",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "bgcolor": "rgba(200,200,200,0.9)",
                "bordercolor": "black",
                "borderwidth": 1
            },
            {
                "buttons": focus_buttons,
                "direction": "down",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.95,
                "xanchor": "right",
                "y": 1.15,
                "yanchor": "top",
                "bgcolor": "rgba(200,200,200,0.9)",
                "bordercolor": "black",
                "borderwidth": 1
            }
        ]
    )
    
    # Log the figure to wandb
    run_name = args.weights.split('/')[-1].split('.')[0]+'_test'
    wandb.init(entity=args.wandb_entity,project=args.wandb_project, name=run_name)
    wandb.log({"confusion_matrix": wandb.Plotly(fig), "test_accuracy":np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)})
    wandb.finish()

    #print("Final Test accuracy:", cor/(cor+incor))

if __name__ == '__main__':
    main()
