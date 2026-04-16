from pathlib import Path


class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    DATA_ROOT = Path(r"C:\Users\91630\Desktop\projects\xai\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1") # ← change to your path
    CSV_PATH    = DATA_ROOT / "ptbxl_database.csv"
    SCP_PATH    = DATA_ROOT / "scp_statements.csv"
    OUTPUT_DIR  = Path("outputs")
    MODEL_DIR   = OUTPUT_DIR / "models"
    PLOT_DIR    = OUTPUT_DIR / "plots"
    LOG_DIR     = OUTPUT_DIR / "logs"

    # ── Data ───────────────────────────────────────────────────────────────
    SAMPLING_RATE   = 100           # 100 → records100 | 500 → records500
    NUM_LEADS       = 12
    SEQ_LEN         = 1000          # 10s × 100 Hz  (use 5000 for 500 Hz)
    LABEL_THRESHOLD = 50.0          # min confidence % to count a label

    # ── Labels ─────────────────────────────────────────────────────────────
    NUM_CLASSES        = 71
    NUM_SUPERCLASSES   = 7
    SUPERCLASSES       = ["NORM", "MI", "STTC", "CD", "HYP", "FORM", "RHYTHM"]

    # ── PatchTST ───────────────────────────────────────────────────────────
    PATCH_LEN   = 50     # length of each patch
    STRIDE      = 25     # stride between patches  → 39 patches per lead
    D_MODEL     = 256    # transformer hidden dim
    N_HEADS     = 8      # attention heads
    N_LAYERS    = 6      # transformer encoder layers
    D_FF        = 256    # feed-forward dim
    DROPOUT     = 0.1

    # ── Training ───────────────────────────────────────────────────────────
    BATCH_SIZE              = 32
    EPOCHS                  = 50
    LR                      = 1e-4
    WEIGHT_DECAY            = 1e-5
    EARLY_STOPPING_PATIENCE = 10

    # ── PTB-XL recommended train/val split (fold column) ───────────────────
    TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
    VAL_FOLD    = 9
    TEST_FOLD   = 10

    # ── Device & reproducibility ───────────────────────────────────────────
    DEVICE = "cuda"   # "cpu" if no GPU
    SEED   = 42


cfg = Config()
