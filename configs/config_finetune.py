from datetime import datetime
import torch
import os

# --- Paths and Directories ---
USER_ROOT_DIR = "/sci/labs/arieljaffe/dan.abergel1"
BASE_DATA_PATH = os.path.join(USER_ROOT_DIR,"adni_data_iclr")
BASE_RESULTS_PATH = "results/"
# Path to the best checkpoint from the pretraining run
PRETRAINED_CHECKPOINT_PATH = "/sci/labs/arieljaffe/dan.abergel1/tracker_adni/pretraining_runs/0dfc562c-088c-4de5-a9a8-be7b5a50c305/model.pt"
print("pretrained model path is", BASE_DATA_PATH)
FINETUNE_MODEL_DIR = os.path.join(USER_ROOT_DIR,"results/finetuning_runs/")
print("results path is", FINETUNE_MODEL_DIR)


# --- Data Settings ---
# These should match the pretraining config to ensure data consistency
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
SEED = 44
WINDOW_SIZE = 10
REMOVE_TOP_K_STD = 1

# --- Fine-tuning Task Configuration ---
# Example: Predicting cognitive decline at 1-year horizon
# FINETUNE_TASK = "degradation_binary_1year"
FINETUNE_TASK = "Sex_Binary"
TASKS_TYPES = {
    FINETUNE_TASK: "binary",
}

OUTPUT_DIMS = {FINETUNE_TASK: 1}
CHOSEN_LABELS = [FINETUNE_TASK]
TASK_WEIGHTS = {FINETUNE_TASK: 1.0}

# --- Fine-tuning Head & Training Hyperparameters ---
# The input dimension must match the flattened bottleneck dimension from the pretrained model
# (MERGE_PATCHES * EMBEDDING_DIM from pretrain config = 10 * 128 = 1280)
HEAD_INPUT_DIM = 1280
HEAD_P_DROPOUT = 0.2
LR = 3e-5  #
NUM_EPOCHS = 10
BATCH_SIZE = 32
OPTIMIZER_WEIGHT_DECAY = 5e-6

# --- Runtime and Logging ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = 1
START_DATE = datetime.now().strftime("%d/%m/%Y")

# --- Checkpointing ---
CHECKPOINT_PERCENTAGE = 0.0  # Only save at the end
BEST_METRIC_NAME = "precision"  # Monitor F1-score for best model


# # --- Task and Loss Configuration ---

TASKS = [
    FINETUNE_TASK,
]

