import os

import torch
import torch.optim as optim

from ..configs import config_finetune as config

from ..models.heads import PredictionHead
# from ..models.ag_vit import TransformerAutoEncoder
from ..training.losses import get_loss_fns, calculate_balanced_weights
from ..training.trainer import FineTuner
from ..utils.metrics import MetricsTracker
from ..utils.tracking import RunTracker
from ..data.prepare_data import prepare_dataloaders
from ..utils.general import set_seed
from torch.serialization import add_safe_globals
from ..models.ag_vit import TransformerAutoEncoder,PatchSelection
import json
from pathlib import Path

def make_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)

add_safe_globals([
    TransformerAutoEncoder,
    PatchSelection
])
def main():
    # --- Setup ---
    set_seed(config.SEED, config.DEVICE)

    device = torch.device(config.DEVICE)
    # --- Load Data ---

    data_components = prepare_dataloaders(config, stage="finetune")
    # take from here the labels
    train_dataloader = data_components["train_dataloader"]

    val_dataloader = data_components["val_dataloader"]
    # --- Load Pretrained Model ---
    print(f"Loading pretrained model from: {config.PRETRAINED_CHECKPOINT_PATH}")
    pretrained_model = torch.load(
        config.PRETRAINED_CHECKPOINT_PATH, map_location=device,weights_only = False
    )
    print("PRETRAINED MODEL =====> ",pretrained_model)
    def count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    total, trainable = count_params(pretrained_model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")


    pretrained_model.eval()
    # --- Initialize Prediction Head ---

    prediction_head = PredictionHead(
        embedding_dim=config.HEAD_INPUT_DIM,
        output_dims=config.OUTPUT_DIMS,
        p_dropout=config.HEAD_P_DROPOUT,
    ).to(device)
    # --- Setup Training Components ---
    optimizer = optim.AdamW(
        prediction_head.parameters(),
        lr=config.LR,
        weight_decay=config.OPTIMIZER_WEIGHT_DECAY,
    )
    imageID_to_labels = data_components["imageID_to_labels"]
    for k, v in imageID_to_labels.items():
        if "Age_in_Yrs" in v:
            v["Age_in_Yrs"] = float(v["Age_in_Yrs"])
    class_weights = calculate_balanced_weights(imageID_to_labels, config.TASKS_TYPES, device)
    loss_fns = get_loss_fns(config, class_weights=class_weights)
    metrics_tracker_tr = MetricsTracker(config.TASKS_TYPES)
    metrics_tracker_val = MetricsTracker(config.TASKS_TYPES)

    tracker = RunTracker(base_dir=config.FINETUNE_MODEL_DIR)

    # Clean and convert hyperparams
    clean_hyperparams = {}
    for k, v in vars(config).items():
        clean_hyperparams[k] = make_json_safe(v)

    clean_hyperparams["chosen_labels"] = config.FINETUNE_TASK
    clean_hyperparams["num_epochs"] = config.NUM_EPOCHS

    # --- Initialize and Run FineTuner ---
    finetuner = FineTuner(
        model=pretrained_model,
        prediction_head=prediction_head,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        tracker=tracker,
        hyperparams=clean_hyperparams,
        metrics_tracker_tr=metrics_tracker_tr,
        metrics_tracker_test=metrics_tracker_val,
        loss_fns=loss_fns,
        freeze_model=True,
    )

    print("Starting fine-tuning...")
    finetuner.train()
    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
