import optuna
from pathlib import Path
import pandas as pd

# -------------------------------------------------
# Imports projet
# -------------------------------------------------
from ..configs import config_pretrain as config
from .run_pretraining import main   # adapte si besoin (module vs script)


# -------------------------------------------------
# Utils
# -------------------------------------------------
def load_last_val_loss(run_dir: Path):
    metrics_dir = run_dir / "metrics"
    test_csv = metrics_dir / "test.csv"

    if not test_csv.exists():
        raise RuntimeError(f"[Optuna] Missing metrics file: {test_csv}")

    df = pd.read_csv(test_csv)

    if "Reconstruction_loss" not in df.columns:
        raise RuntimeError("[Optuna] Column Reconstruction_loss not found in test.csv")

    return float(df["Reconstruction_loss"].iloc[-1])


# -------------------------------------------------
# Optuna objective
# -------------------------------------------------
def objective(trial):

    print("\n" + "=" * 90)
    print(f"[Optuna] 🚀 Starting trial {trial.number}")
    print("=" * 90)

    # -------- 1. Sample hyperparameters --------
    config.LR = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    config.EMBEDDING_DIM = trial.suggest_categorical(
        "embedding_dim", [64, 128, 256]
    )
    config.NUM_HEADS = trial.suggest_categorical(
        "num_heads", [2, 4, 8]
    )
    config.P_DROPOUT = trial.suggest_float(
        "dropout", 0.0, 0.3
    )
    config.OPTIMIZER_WEIGHT_DECAY = trial.suggest_float(
        "weight_decay", 1e-6, 1e-3, log=True
    )

    print(
        f"[Optuna][Trial {trial.number}] Hyperparameters:\n"
        f"  LR            = {config.LR:.2e}\n"
        f"  EMBEDDING_DIM = {config.EMBEDDING_DIM}\n"
        f"  NUM_HEADS     = {config.NUM_HEADS}\n"
        f"  DROPOUT       = {config.P_DROPOUT:.2f}\n"
        f"  WEIGHT_DECAY  = {config.OPTIMIZER_WEIGHT_DECAY:.2e}"
    )

    # -------- 2. Contraintes architecture --------
    if config.EMBEDDING_DIM % config.NUM_HEADS != 0:
        print(
            f"[Optuna][Trial {trial.number}] ❌ Pruned "
            f"(embedding_dim % num_heads != 0)"
        )
        raise optuna.TrialPruned()

    # -------- 3. Epochs réduites pour Optuna --------
    config.NUM_EPOCHS = 10
    print(f"[Optuna][Trial {trial.number}] NUM_EPOCHS set to {config.NUM_EPOCHS}")

    # -------- 4. Lancer le training --------
    print(f"[Optuna][Trial {trial.number}] ▶️ Launching training...")
    main()
    print(f"[Optuna][Trial {trial.number}] ✅ Training finished")

    # -------- 5. Trouver le dernier run --------
    run_dirs = sorted(
        config.PRETRAINED_MODEL_DIR.glob("*"),
        key=lambda p: p.stat().st_mtime,
    )

    if not run_dirs:
        raise RuntimeError("[Optuna] No run directory found")

    last_run = run_dirs[-1]
    print(f"[Optuna][Trial {trial.number}] Using run: {last_run.name}")

    # -------- 6. Lire la métrique --------
    val_loss = load_last_val_loss(last_run)

    print(
        f"[Optuna][Trial {trial.number}] 🎯 "
        f"Final Reconstruction_loss = {val_loss:.6f}"
    )

    print("=" * 90)
    print(f"[Optuna] ✅ Trial {trial.number} completed")
    print("=" * 90)

    return val_loss


# -------------------------------------------------
# Main Optuna loop
# -------------------------------------------------
if __name__ == "__main__":

    print("\n" + "#" * 100)
    print("# Starting Optuna hyperparameter optimization")
    print("#" * 100)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=44),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(objective, n_trials=12)

    print("\n" + "#" * 100)
    print("# Optuna finished")
    print("#" * 100)

    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.6f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")