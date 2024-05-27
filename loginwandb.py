import os
import wandb

key_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'wandb_key.txt')

try:
    with open(key_file_path, 'r') as file:
        api_key = file.read().strip()
except FileNotFoundError:
    raise Exception(f"API key file not found at {key_file_path}")

wandb.login(key=api_key)

# wandb.login(key=api_key)

# # # Example wandb initialization
# wandb.init(project='prototype_segmentation-test5')

# # Log some metrics
# wandb.log({'phase': 0, 'accuracy': 0.9})

# wandb.log({'phase': 0, 'loss': 0.1})

# wandb.log({'phase': 1, 'accuracy': 0.6})

# wandb.log({'phase': 1, 'loss': 0.15})

# # Finish the run
# wandb.finish()