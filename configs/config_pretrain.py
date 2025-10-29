from datetime import datetime
import torch

# --- Paths and Directories ---
BASE_DATA_PATH = "/sci/labs/arieljaffe/dan.abergel1/model_test"
BASE_RESULTS_PATH = "/results/"
PRETRAINED_MODEL_DIR = "/tracker/pretraining_runs/"
ATLAS = "schaefer200"

# --- Data Settings ---
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
SEED = 44
WINDOW_SIZE = 10
REMOVE_TOP_K_STD = 1

# --- Model Architecture ---
PATCH_SIZE = (6, 6, 6)
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
NUM_CLS_TOKENS = 1
P_DROPOUT = 0.1
REDUCED_PATCHES_FACTOR_PERCENT = 0.1
REDUCE_TIME_FACTOR_PERCENT = 0.2
MERGE_PATCHES = 10
USE_PATCH_MERGER = True
CUSTOM_RECON_BOOL = True

# --- Training Configuration ---
TASKS = []
TASK_WEIGHTS = {"Reconstruction": 1.0}
TASKS_TYPES = {"Reconstruction": "regression"}
CHOSEN_LABELS = ["Reconstruction"]

LR = 5e-4
NUM_EPOCHS = 55
BATCH_SIZE = 8
OPTIMIZER_WEIGHT_DECAY = 0.0005
MAX_NORM = 1.0
WARMUP_STEPS_PERCENT = 0.1

# # --- Task and Loss Configuration ---

# TASKS = [
#     "Sex_Binary",
#     "MMSE_Binary",
#     "CDR_Binary",
#     "FAQ_Binary",
#     "Age_Category",
#     "GDSCALE_Category",
#     "CDR_Category",
#     "CDMEMORY",
#     "CDRSB",
#     "degradation_binary_1year",
#     "degradation_binary_2years",
#     "degradation_binary_3years",
# ]

# # Weights for each task's loss
# TASK_WEIGHTS = {
#     "Sex_Binary": 1,
#     "MMSE_Binary": 1,
#     "CDR_Binary": 1,
#     "FAQ_Binary": 1,
#     "Age_Category": 1,
#     "GDSCALE_Category": 1,
#     "CDR_Category": 1,
#     "CDMEMORY": 1,
#     "CDRSB": 1,
#     "Reconstruction": 1,
# }

# # Mapping of task names to their type ('binary', 'categorical', 'regression')
# TASKS_TYPES = {
#     "Sex_Binary": "binary",
#     "MMSE_Binary": "binary",
#     "CDR_Binary": "binary",
#     "FAQ_Binary": "binary",
#     "Age_Category": "categorical",
#     "GDSCALE_Category": "categorical",
#     "CDR_Category": "categorical",
#     "CDMEMORY": "categorical",
#     "CDRSB": "categorical",
#     "Reconstruction": "regression",
#     "degradation_binary_1year": "binary",
#     "degradation_binary_2years": "binary",
#     "degradation_binary_3years": "binary",
# }


# CATEGORY_TO_DIM = {
#     "Sex_Binary": 1,
#     "MMSE_Binary": 1,
#     "CDR_Binary": 1,
#     "FAQ_Binary": 1,
#     "Age_Category": 4,
#     "GDSCALE_Category": 4,
#     "CDR_Category": 4,
#     "CDMEMORY": 4,
#     "CDRSB": 19,
#     "degradation_binary_1year": 1,
#     "degradation_binary_2years": 1,
#     "degradation_binary_3years": 1,
# }

# --- Runtime and Logging ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = 1  # 0: silent, 1: epoch tqdm, 2: epoch and batch tqdm
START_DATE = datetime.now().strftime("%d/%m/%Y")
TRACK_GRAD = True

# --- Checkpointing and Plotting ---
CHECKPOINT_PERCENTAGE = (
    0.0  # Percentage of epochs to save a checkpoint, 0 saves only at the end
)
CHECKPOINT_TASK = "Reconstruction"  # Task to monitor for best model saving
BEST_METRIC_NAME = "loss"  # Metric to monitor ('loss', 'accuracy', 'f1', etc.)
PLOT_PERCENTAGE = 0.25  # Percentage of epochs to generate and save plots
MAX_ROWS_FOR_PLOT = 2
