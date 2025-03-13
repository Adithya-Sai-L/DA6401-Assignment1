import argparse
import wandb

from keras.datasets import fashion_mnist, mnist
import numpy as np

parser = argparse.ArgumentParser(description='Accept Command line of wandb_project, wandb_entity, dataset')
parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')    
parser.add_argument('-we','--wandb_entity', type=str, default='myname',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')  
parser.add_argument('-d','--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help="Choose dataset from ['mnist', 'fashion_mnist']")
parser.add_argument('-s','--seed', type=int, default=42, help="Seed used to initialize random number generators.")

args = parser.parse_args()

wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
wandb.run.name = f"plot_images_{args.dataset}"
wandb.run.save()


if args.dataset == "fashion_mnist":
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
elif args.dataset == "mnist":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
else:
    raise ValueError("Invalid dataset")

np.random.seed(args.seed)
imgs = []
for i in range(10):
    imgs.append(train_images[np.random.choice(np.where(train_labels == i)[0])])

wandb.log({"train_images": [wandb.Image(img, caption=class_names[i]) for i, img in enumerate(imgs)]})

imgs = []
for i in range(10):
    imgs.append(test_images[np.random.choice(np.where(test_labels == i)[0])])

wandb.log({"test_images": [wandb.Image(img, caption=class_names[i]) for i, img in enumerate(imgs)]})