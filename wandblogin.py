import wandb

#WANDB_API_KEY = os.environ['WANDB_API_KEY']
keyFile = open('wandb.key', 'r')
WANDB_API_KEY = keyFile.readline().rstrip()
wandb.login(key=WANDB_API_KEY)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
)