# -*- coding: utf-8 -*-
"""
Novo treinamento - Revisão 2 - MODO COMPLETO

Objetivo do modo completo:
1) Usar apenas as 3 classes geológicas: Sandstone, Limestone e Shale.
2) Remover a classe Others/COCO da análise principal.
3) Corrigir o redimensionamento, evitando nearest-neighbor.
   - Este script usa PIL Image.Resampling.BOX, uma interpolação baseada em área
     apropriada para downsampling.
4) Preparar o fluxo grouped 3-fold cross-validation.
5) Rodar todos os 17 modelos em grouped 3-fold cross-validation, com 20 épocas.

Rodada completa:
17 modelos × 3 folds × 20 épocas = 51 treinamentos.
"""

# =============================================================================
# 0. Configuração de DLLs CUDA/cuDNN antes de importar TensorFlow
# =============================================================================

import os
import sys
from pathlib import Path

# Mantém os handles ativos para que o Windows não descarte os diretórios de DLL.
_DLL_HANDLES = []

def add_nvidia_dll_dirs_from_env():
    """
    Adiciona automaticamente ao PATH as pastas das DLLs NVIDIA instaladas via pip
    dentro do ambiente Python atual. Isso ajuda quando o TensorFlow 2.10 encontra
    o driver NVIDIA, mas não encontra cudart/cublas/cudnn.
    """
    env_prefix = Path(sys.prefix)
    nvidia_base = env_prefix / "Lib" / "site-packages" / "nvidia"

    subdirs = [
        "cuda_runtime/bin",
        "cublas/bin",
        "cudnn/bin",
        "cufft/bin",
        "curand/bin",
        "cusolver/bin",
        "cusparse/bin",
        "cuda_nvrtc/bin",
    ]

    for sub in subdirs:
        d = nvidia_base / sub
        if d.exists():
            d_str = str(d)
            try:
                _DLL_HANDLES.append(os.add_dll_directory(d_str))
            except Exception:
                pass
            os.environ["PATH"] = d_str + os.pathsep + os.environ.get("PATH", "")

add_nvidia_dll_dirs_from_env()

# =============================================================================
# 1. Importações
# =============================================================================

import gc
import json
import time
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, ResNet101, ResNet152,
    DenseNet121, DenseNet169, DenseNet201,
    InceptionV3, InceptionResNetV2,
    VGG16, VGG19,
    MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    NASNetMobile, NASNetLarge,
    Xception
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras import backend as K

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# 2. Configurações principais
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Caminho do dataset original usado anteriormente.
DATASET_DIR = Path(r"C:\IA Treinamento\Imagens para treinar")

# Pasta solicitada para salvar todas as saídas do novo treinamento.
OUTPUT_DIR = Path(
    r"E:\OneDrive - Universidade Federal de Pernambuco\0009 Papers em elaboração\032-IA\Versão 9 - Earth Science Informatics\Revisão 2\Novo treinamento"
)

# Modo completo: apenas 1 modelo, 1 fold e poucas épocas.
TEST_MODE = False

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
N_SPLITS = 3

# Para completo rápido. Na rodada completa, mudar para 20.
EPOCHS = 2 if TEST_MODE else 20

# Exigir GPU para evitar treinamento acidental em CPU.
REQUIRE_GPU = True

# No modo completo, salvar modelo não é necessário.
SAVE_MODELS = False

# Permite retomar caso a execução seja interrompida.
RESUME = True

# Classes geológicas usadas na nova análise principal.
# A ordem será usada nas tabelas/matrizes.
CLASS_NAMES = ["Limestone", "Sandstone", "Shale"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Mapeia possíveis nomes de pastas do dataset para rótulos canônicos.
FOLDER_TO_CLASS = {
    "limestone": "Limestone",
    "limestones": "Limestone",
    "carbonate": "Limestone",
    "carbonates": "Limestone",

    "sandstone": "Sandstone",
    "sandstones": "Sandstone",

    "shale": "Shale",
    "shales": "Shale",
}

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Modelos: no completo roda só DenseNet121.
# A rodada completa usará todos.
ALL_MODELS = [
    ("ResNet50", ResNet50),
    ("ResNet101", ResNet101),
    ("ResNet152", ResNet152),
    ("DenseNet121", DenseNet121),
    ("DenseNet169", DenseNet169),
    ("DenseNet201", DenseNet201),
    ("InceptionV3", InceptionV3),
    ("InceptionResNetV2", InceptionResNetV2),
    ("VGG16", VGG16),
    ("VGG19", VGG19),
    ("MobileNet", MobileNet),
    ("MobileNetV2", MobileNetV2),
    ("MobileNetV3Small", MobileNetV3Small),
    ("MobileNetV3Large", MobileNetV3Large),
    ("NASNetMobile", NASNetMobile),
    ("NASNetLarge", NASNetLarge),
    ("Xception", Xception),
]

MODELS_TO_RUN = [("DenseNet121", DenseNet121)] if TEST_MODE else ALL_MODELS
FOLDS_TO_RUN = [1] if TEST_MODE else [1, 2, 3]

# =============================================================================
# 3. Preparação inicial
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Novo treinamento - Revisão 2")
print("MODO COMPLETO" if TEST_MODE else "MODO COMPLETO")
print("=" * 80)
print(f"Dataset: {DATASET_DIR}")
print(f"Saídas:  {OUTPUT_DIR}")
print(f"Classes: {CLASS_NAMES}")
print(f"IMG_SIZE: {IMG_SIZE}")
print(f"EPOCHS: {EPOCHS}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print("=" * 80)

# Verificação de GPU.
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs detected: {len(gpus)}")
for gpu in gpus:
    print(f" - {gpu}")

if REQUIRE_GPU and len(gpus) == 0:
    raise RuntimeError(
        "Nenhuma GPU foi detectada pelo TensorFlow. "
        "Pare aqui e corrija o ambiente CUDA/cuDNN antes de treinar."
    )

# Crescimento dinâmico de memória na GPU.
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# =============================================================================
# 4. Indexação do dataset com apenas 3 classes
# =============================================================================

def canonical_class_from_folder(folder_name: str):
    return FOLDER_TO_CLASS.get(folder_name.lower().strip(), None)

def build_dataset_index(dataset_dir: Path) -> pd.DataFrame:
    """
    Lê o diretório do dataset e cria um índice apenas com as classes geológicas.
    Ignora automaticamente Others/Outros/COCO ou qualquer outra pasta não mapeada.
    """
    rows = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_dir}")

    top_folders = [p for p in dataset_dir.iterdir() if p.is_dir()]
    if not top_folders:
        raise RuntimeError(f"Nenhuma subpasta encontrada em: {dataset_dir}")

    for class_folder in top_folders:
        class_label = canonical_class_from_folder(class_folder.name)

        # Ignora Others/Outros/COCO e qualquer pasta não reconhecida.
        if class_label is None:
            print(f"Ignorando pasta não geológica/não mapeada: {class_folder.name}")
            continue

        for img_path in class_folder.rglob("*"):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rel_parts = img_path.relative_to(dataset_dir).parts

            # Grupo físico/amostra: subpasta imediatamente dentro da classe.
            # Ex.: Limestone/Sample01/img.tif => Sample01.
            # Se não houver subpasta, usa o nome do arquivo como fallback.
            if len(rel_parts) >= 3:
                sample_id = rel_parts[1]
            else:
                sample_id = img_path.stem

            group_id = f"{class_label}__{sample_id}"

            rows.append({
                "path": str(img_path),
                "relative_path": str(img_path.relative_to(dataset_dir)),
                "folder_class": class_folder.name,
                "label": class_label,
                "class_index": CLASS_TO_INDEX[class_label],
                "sample_id": sample_id,
                "group_id": group_id,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "Nenhuma imagem das 3 classes geológicas foi encontrada. "
            "Verifique os nomes das pastas e o caminho do dataset."
        )

    df = df.sort_values(["label", "group_id", "relative_path"]).reset_index(drop=True)
    return df

df = build_dataset_index(DATASET_DIR)

dataset_index_path = OUTPUT_DIR / "01_dataset_index_3classes_FULL.csv"
df.to_csv(dataset_index_path, index=False, encoding="utf-8-sig")

print("\nResumo do dataset indexado:")
print(df["label"].value_counts())
print("\nResumo de grupos físicos por classe:")
print(df.groupby("label")["group_id"].nunique())

group_summary = (
    df.groupby(["label", "group_id"])
      .size()
      .reset_index(name="n_slices")
      .sort_values(["label", "group_id"])
)
group_summary_path = OUTPUT_DIR / "02_physical_group_summary_3classes_FULL.csv"
group_summary.to_csv(group_summary_path, index=False, encoding="utf-8-sig")

# =============================================================================
# 5. Split grouped 3-fold
# =============================================================================

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_assignments = []
folds = {}

X = df["path"].values
y = df["class_index"].values
groups = df["group_id"].values

for fold_number, (trainval_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
    trainval_df = df.iloc[trainval_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Split de validação dentro do trainval, mantendo grupos.
    # Tenta algumas sementes para evitar validação sem alguma classe.
    val_ok = False
    last_error = None
    for trial_seed in range(SEED, SEED + 100):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=trial_seed)
        try:
            tr_idx, val_idx = next(
                gss.split(
                    trainval_df["path"].values,
                    trainval_df["class_index"].values,
                    groups=trainval_df["group_id"].values,
                )
            )
            train_df = trainval_df.iloc[tr_idx].copy().reset_index(drop=True)
            val_df = trainval_df.iloc[val_idx].copy().reset_index(drop=True)

            if set(train_df["class_index"].unique()) == set(range(NUM_CLASSES)) and \
               set(val_df["class_index"].unique()) == set(range(NUM_CLASSES)):
                val_ok = True
                break
        except Exception as e:
            last_error = e

    if not val_ok:
        raise RuntimeError(
            f"Não foi possível criar split treino/validação com todas as classes no fold {fold_number}. "
            f"Último erro: {last_error}"
        )

    folds[fold_number] = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        temp = split_df[["path", "relative_path", "label", "class_index", "sample_id", "group_id"]].copy()
        temp["fold"] = fold_number
        temp["split"] = split_name
        fold_assignments.append(temp)

fold_assignments_df = pd.concat(fold_assignments, ignore_index=True)
fold_assignments_path = OUTPUT_DIR / "03_fold_split_assignments_FULL.csv"
fold_assignments_df.to_csv(fold_assignments_path, index=False, encoding="utf-8-sig")

print("\nDistribuição por fold/split/classe:")
dist = (
    fold_assignments_df
    .groupby(["fold", "split", "label"])
    .size()
    .reset_index(name="n_slices")
)
print(dist)

dist.to_csv(OUTPUT_DIR / "04_fold_class_distribution_FULL.csv", index=False, encoding="utf-8-sig")

# =============================================================================
# 6. Gerador customizado com redimensionamento BOX/area-like
# =============================================================================

def pil_resampling_box():
    """
    Compatibilidade com versões diferentes do Pillow.
    BOX é apropriado para downsampling por média local/area-like.
    """
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BOX
    return Image.BOX

RESAMPLE_METHOD = pil_resampling_box()

def load_image_area_like(path: str) -> np.ndarray:
    """
    Lê uma imagem, converte para RGB e redimensiona para 224x224 usando BOX.
    BOX é um filtro de média local adequado para reduzir imagens grandes,
    evitando nearest-neighbor e reduzindo aliasing.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, resample=RESAMPLE_METHOD)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

class RockSequence(Sequence):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        batch_size: int,
        num_classes: int,
        shuffle: bool = False,
        augment: bool = False,
        seed: int = 42,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        self.indices = np.arange(len(self.df))

        if self.augment:
            self.augmenter = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode="nearest",
            )
        else:
            self.augmenter = None

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.indices)
            self.seed += 1

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        images = np.zeros((len(batch_df), IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.float32)
        labels = np.zeros((len(batch_df),), dtype=np.int32)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            img = load_image_area_like(row["path"])

            if self.augmenter is not None:
                img = self.augmenter.random_transform(img)
                img = np.clip(img, 0.0, 1.0)

            images[i] = img
            labels[i] = int(row["class_index"])

        y = to_categorical(labels, num_classes=self.num_classes)
        return images, y

# =============================================================================
# 7. Métricas e utilidades
# =============================================================================

def mean_per_class_accuracy(y_true, y_pred, num_classes):
    values = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for c in range(num_classes):
        mask = y_true == c
        if np.sum(mask) > 0:
            values.append(np.mean(y_pred[mask] == c))

    return float(np.mean(values)) if values else 0.0

def build_model(model_name, model_fn):
    """
    Cria modelo com backbone ImageNet congelado e head comum.
    A saída tem 3 classes.
    """
    kwargs = dict(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3))

    # Garante consistência: não usar preprocessing interno nos MobileNetV3,
    # pois já normalizamos manualmente para [0, 1].
    if model_name in ["MobileNetV3Small", "MobileNetV3Large"]:
        kwargs["include_preprocessing"] = False

    base_model = model_fn(**kwargs)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def plot_learning_curve(history_df, metric, out_path, title):
    plt.figure()
    if metric in history_df.columns:
        plt.plot(history_df[metric], label=f"Train {metric}")
    val_metric = f"val_{metric}"
    if val_metric in history_df.columns:
        plt.plot(history_df[val_metric], label=f"Validation {metric}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm_norm, class_names, out_path, title):
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.3f}", ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def majority_vote_specimen_metrics(pred_df: pd.DataFrame):
    """
    Agrega predições por group_id usando majority vote.
    Retorna dataframe por espécime/grupo e métricas agregadas.
    """
    rows = []
    for group_id, g in pred_df.groupby("group_id"):
        true_label = g["true_label"].iloc[0]
        true_index = int(g["true_index"].iloc[0])

        pred_counts = Counter(g["pred_index"].astype(int).tolist())
        pred_index = pred_counts.most_common(1)[0][0]
        pred_label = CLASS_NAMES[pred_index]

        rows.append({
            "group_id": group_id,
            "true_label": true_label,
            "true_index": true_index,
            "pred_label": pred_label,
            "pred_index": pred_index,
            "n_slices": len(g),
            "slice_accuracy_within_group": float(np.mean(g["true_index"].values == g["pred_index"].values)),
        })

    group_pred = pd.DataFrame(rows)

    y_true_g = group_pred["true_index"].values
    y_pred_g = group_pred["pred_index"].values

    metrics = {
        "specimen_accuracy": accuracy_score(y_true_g, y_pred_g),
        "specimen_precision_weighted": precision_score(y_true_g, y_pred_g, average="weighted", zero_division=0),
        "specimen_recall_weighted": recall_score(y_true_g, y_pred_g, average="weighted", zero_division=0),
        "specimen_f1_weighted": f1_score(y_true_g, y_pred_g, average="weighted", zero_division=0),
        "specimen_mpca": mean_per_class_accuracy(y_true_g, y_pred_g, NUM_CLASSES),
        "n_test_groups": len(group_pred),
    }

    return group_pred, metrics

# =============================================================================
# 8. Treinamento em modo completo
# =============================================================================

results_path = OUTPUT_DIR / "results_by_model_and_fold_FULL.csv"

if RESUME and results_path.exists():
    try:
        existing_results = pd.read_csv(results_path)
        all_results = existing_results.to_dict("records")
        completed_jobs = set((str(r["model"]), int(r["fold"])) for _, r in existing_results.iterrows())
        print(f"Retomada ativada: {len(completed_jobs)} combinações modelo/fold já concluídas serão puladas.")
    except Exception as e:
        print(f"Não foi possível ler resultados existentes para retomada: {e}")
        all_results = []
        completed_jobs = set()
else:
    all_results = []
    completed_jobs = set()

run_metadata = {
    "test_mode": TEST_MODE,
    "dataset_dir": str(DATASET_DIR),
    "output_dir": str(OUTPUT_DIR),
    "classes": CLASS_NAMES,
    "num_classes": NUM_CLASSES,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "n_splits": N_SPLITS,
    "folds_to_run": FOLDS_TO_RUN,
    "models_to_run": [m[0] for m in MODELS_TO_RUN],
    "resize_method": "PIL Image.Resampling.BOX (area-like downsampling)",
    "tensorflow_version": tf.__version__,
    "gpu_devices": [str(g) for g in gpus],
    "created_at_unix": time.time(),
    "resume": RESUME,
}

with open(OUTPUT_DIR / "00_run_metadata_FULL.json", "w", encoding="utf-8") as f:
    json.dump(run_metadata, f, indent=2, ensure_ascii=False)

for fold_number in FOLDS_TO_RUN:
    split = folds[fold_number]
    train_df = split["train"]
    val_df = split["val"]
    test_df = split["test"]

    print("\n" + "=" * 80)
    print(f"FOLD {fold_number}/{N_SPLITS}")
    print("=" * 80)
    print("Train:")
    print(train_df["label"].value_counts())
    print("Validation:")
    print(val_df["label"].value_counts())
    print("Test:")
    print(test_df["label"].value_counts())

    train_gen = RockSequence(
        train_df,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=True,
        augment=True,
        seed=SEED + fold_number,
    )

    val_gen = RockSequence(
        val_df,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=False,
        augment=False,
        seed=SEED,
    )

    test_gen = RockSequence(
        test_df,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=False,
        augment=False,
        seed=SEED,
    )

    for model_name, model_fn in MODELS_TO_RUN:
        if RESUME and (model_name, int(fold_number)) in completed_jobs:
            print("\n" + "-" * 80)
            print(f"Pulando {model_name} | Fold {fold_number}/{N_SPLITS}: já concluído")
            print("-" * 80)
            continue

        print("\n" + "-" * 80)
        print(f"Training {model_name} | Fold {fold_number}/{N_SPLITS}")
        print("-" * 80)

        model = build_model(model_name, model_fn)

        start_time = time.time()
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            verbose=1,
        )
        training_time_s = time.time() - start_time

        # Salva histórico.
        suffix = f"{model_name}_fold{fold_number}_FULL"
        history_df = pd.DataFrame(history.history)
        history_path = OUTPUT_DIR / f"training_log_{suffix}.csv"
        history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

        plot_learning_curve(
            history_df,
            metric="accuracy",
            out_path=OUTPUT_DIR / f"learning_curve_accuracy_{suffix}.png",
            title=f"{model_name} Fold {fold_number} Accuracy",
        )

        plot_learning_curve(
            history_df,
            metric="loss",
            out_path=OUTPUT_DIR / f"learning_curve_loss_{suffix}.png",
            title=f"{model_name} Fold {fold_number} Loss",
        )

        # Avaliação/predição.
        eval_loss, eval_accuracy = model.evaluate(test_gen, verbose=1)
        y_prob = model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = test_df["class_index"].astype(int).values

        # Métricas slice-level.
        slice_accuracy = accuracy_score(y_true, y_pred)
        slice_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        slice_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        slice_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        slice_mpca = mean_per_class_accuracy(y_true, y_pred, NUM_CLASSES)

        # Relatório classificação.
        report_txt = classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )

        with open(OUTPUT_DIR / f"classification_report_{suffix}.txt", "w", encoding="utf-8") as f:
            f.write(report_txt)

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        pd.DataFrame(report_dict).transpose().to_csv(
            OUTPUT_DIR / f"classification_report_{suffix}.csv",
            encoding="utf-8-sig",
        )

        # Matriz de confusão.
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        cm_df.to_csv(OUTPUT_DIR / f"confusion_matrix_{suffix}.csv", encoding="utf-8-sig")

        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        cm_norm_df = pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES)
        cm_norm_df.to_csv(OUTPUT_DIR / f"confusion_matrix_normalized_{suffix}.csv", encoding="utf-8-sig")

        plot_confusion_matrix(
            cm_norm,
            class_names=CLASS_NAMES,
            out_path=OUTPUT_DIR / f"confusion_matrix_normalized_{suffix}.png",
            title=f"{model_name} Fold {fold_number} normalized confusion matrix",
        )

        # Predições por slice.
        pred_df = test_df[["path", "relative_path", "label", "class_index", "sample_id", "group_id"]].copy()
        pred_df = pred_df.rename(columns={"label": "true_label", "class_index": "true_index"})
        pred_df["pred_index"] = y_pred
        pred_df["pred_label"] = [CLASS_NAMES[i] for i in y_pred]

        for idx, class_name in enumerate(CLASS_NAMES):
            pred_df[f"prob_{class_name}"] = y_prob[:, idx]

        predictions_path = OUTPUT_DIR / f"predictions_{suffix}.csv"
        pred_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

        # Agregação por espécime/grupo físico.
        group_pred_df, specimen_metrics = majority_vote_specimen_metrics(pred_df)
        group_pred_df.to_csv(
            OUTPUT_DIR / f"specimen_majority_vote_predictions_{suffix}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        result_row = {
            "mode": "TEST" if TEST_MODE else "FULL",
            "model": model_name,
            "fold": fold_number,
            "epochs": EPOCHS,
            "n_train_slices": len(train_df),
            "n_val_slices": len(val_df),
            "n_test_slices": len(test_df),
            "n_train_groups": train_df["group_id"].nunique(),
            "n_val_groups": val_df["group_id"].nunique(),
            "n_test_groups": test_df["group_id"].nunique(),
            "loss": float(eval_loss),
            "accuracy": float(slice_accuracy),
            "precision_weighted": float(slice_precision),
            "recall_weighted": float(slice_recall),
            "f1_weighted": float(slice_f1),
            "mpca": float(slice_mpca),
            "training_time_s": float(training_time_s),
            "training_time_h": float(training_time_s / 3600.0),
            **specimen_metrics,
        }

        all_results.append(result_row)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_DIR / "results_by_model_and_fold_FULL.csv", index=False, encoding="utf-8-sig")

        print("\nResumo do completo:")
        print(json.dumps(result_row, indent=2, ensure_ascii=False))
        print("\nClassification report:")
        print(report_txt)

        if SAVE_MODELS:
            model.save(OUTPUT_DIR / f"model_{suffix}.h5")

        # Limpa memória antes do próximo modelo.
        K.clear_session()
        del model
        gc.collect()

# =============================================================================
# 9. Resumos finais
# =============================================================================

results_df = pd.DataFrame(all_results)
if not results_df.empty and {"model", "fold"}.issubset(results_df.columns):
    results_df = results_df.drop_duplicates(subset=["model", "fold"], keep="last")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

summary = (
    results_df
    .groupby("model")
    .agg({
        "accuracy": ["mean", "std"],
        "mpca": ["mean", "std"],
        "precision_weighted": ["mean", "std"],
        "recall_weighted": ["mean", "std"],
        "f1_weighted": ["mean", "std"],
        "specimen_accuracy": ["mean", "std"],
        "specimen_mpca": ["mean", "std"],
        "training_time_h": ["mean", "std"],
    })
)

summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
summary = summary.reset_index()

summary_path = OUTPUT_DIR / "summary_mean_std_FULL.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

# Tabela simples para o artigo - apenas completo, não usar ainda no artigo.
article_table = summary.copy()
article_table["Accuracy"] = article_table.apply(
    lambda r: f"{r['accuracy_mean']:.4f} ± {0.0 if pd.isna(r['accuracy_std']) else r['accuracy_std']:.4f}",
    axis=1,
)
article_table["MPCA"] = article_table.apply(
    lambda r: f"{r['mpca_mean']:.4f} ± {0.0 if pd.isna(r['mpca_std']) else r['mpca_std']:.4f}",
    axis=1,
)
article_table["Weighted F1-score"] = article_table.apply(
    lambda r: f"{r['f1_weighted_mean']:.4f} ± {0.0 if pd.isna(r['f1_weighted_std']) else r['f1_weighted_std']:.4f}",
    axis=1,
)
article_table["Specimen-level accuracy"] = article_table.apply(
    lambda r: f"{r['specimen_accuracy_mean']:.4f} ± {0.0 if pd.isna(r['specimen_accuracy_std']) else r['specimen_accuracy_std']:.4f}",
    axis=1,
)

article_table_final = article_table[["model", "Accuracy", "MPCA", "Weighted F1-score", "Specimen-level accuracy"]]
article_table_final.to_csv(OUTPUT_DIR / "article_table_FULL.csv", index=False, encoding="utf-8-sig")

with open(OUTPUT_DIR / "article_table_FULL.txt", "w", encoding="utf-8") as f:
    f.write(article_table_final.to_string(index=False))


# Métricas acumuladas por modelo a partir das predições dos 3 folds.
accumulated_rows = []
for model_name, _ in MODELS_TO_RUN:
    pred_files = sorted(OUTPUT_DIR.glob(f"predictions_{model_name}_fold*_FULL.csv"))
    if not pred_files:
        continue
    pred_all = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True)
    y_true_all = pred_all["true_index"].astype(int).values
    y_pred_all = pred_all["pred_index"].astype(int).values

    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(NUM_CLASSES)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / f"confusion_matrix_accumulated_{model_name}_FULL.csv",
        encoding="utf-8-sig",
    )
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / f"confusion_matrix_accumulated_normalized_{model_name}_FULL.csv",
        encoding="utf-8-sig",
    )
    plot_confusion_matrix(
        cm_norm,
        class_names=CLASS_NAMES,
        out_path=OUTPUT_DIR / f"confusion_matrix_accumulated_normalized_{model_name}_FULL.png",
        title=f"{model_name} accumulated normalized confusion matrix",
    )
    accumulated_rows.append({
        "model": model_name,
        "n_predictions": len(pred_all),
        "accuracy_accumulated": accuracy_score(y_true_all, y_pred_all),
        "precision_weighted_accumulated": precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0),
        "recall_weighted_accumulated": recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0),
        "f1_weighted_accumulated": f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0),
        "mpca_accumulated": mean_per_class_accuracy(y_true_all, y_pred_all, NUM_CLASSES),
    })

accumulated_df = pd.DataFrame(accumulated_rows)
accumulated_df.to_csv(OUTPUT_DIR / "accumulated_metrics_from_all_folds_FULL.csv", index=False, encoding="utf-8-sig")

# Gráfico resumo com F1 médio.
if "f1_weighted_mean" in summary.columns and not summary.empty:
    plt.figure(figsize=(10, 6))
    plt.bar(summary["model"], summary["f1_weighted_mean"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Weighted F1-score, mean across grouped folds")
    plt.title("Grouped 3-fold cross-validation performance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_weighted_f1_mean_FULL.png", dpi=300)
    plt.close()

# Textos-base automáticos para posterior revisão manual.
methods_text = f"""The revised experimental protocol was reformulated as a three-class geological classification problem, including only sandstone, limestone, and shale μCT slices. The artificial non-geological COCO-derived class was removed from the main benchmark. All images were resized to {IMG_SIZE[0]} × {IMG_SIZE[1]} pixels using PIL Image.Resampling.BOX, an area-like downsampling filter selected to avoid nearest-neighbor decimation and reduce aliasing during image reduction. Model evaluation was performed using specimen-level grouped {N_SPLITS}-fold cross-validation. Physical specimen identifiers were used as grouping variables, ensuring that all slices from the same rock specimen were assigned exclusively to a single fold. The 17 CNN architectures were evaluated under the same frozen-backbone transfer-learning protocol, using ImageNet-pretrained weights, a Global Average Pooling layer, a Dense layer with 1024 neurons and ReLU activation, and a final softmax layer with three output neurons."""
with open(OUTPUT_DIR / "methods_text_base_FULL.txt", "w", encoding="utf-8") as f:
    f.write(methods_text)

response_text = """In response to the reviewer’s concerns, we substantially redesigned the experimental evaluation. First, the COCO-derived 'Others' class was removed from the main benchmark, and the task was reformulated as a three-class geological classification problem involving sandstone, limestone, and shale. Second, the preprocessing pipeline was revised to avoid nearest-neighbor downsampling; all images were resized using an area-like BOX resampling filter to reduce aliasing and better preserve local grayscale texture during downsampling. Third, the grouped three-fold cross-validation analysis was extended to all 17 CNN architectures, rather than only a subset of models. Fourth, in addition to slice-level metrics, we computed specimen-level majority-vote metrics to provide an evaluation closer to the physical sampling structure of the dataset. These revisions directly address the concerns regarding artificial metric inflation, destructive preprocessing, incomplete cross-validation, and pseudoreplication-aware evaluation."""
with open(OUTPUT_DIR / "response_text_base_FULL.txt", "w", encoding="utf-8") as f:
    f.write(response_text)

# Também cria Excel se openpyxl estiver disponível.
try:
    with pd.ExcelWriter(OUTPUT_DIR / "new_training_FULL_results.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Dataset index", index=False)
        group_summary.to_excel(writer, sheet_name="Group summary", index=False)
        fold_assignments_df.to_excel(writer, sheet_name="Fold assignments", index=False)
        dist.to_excel(writer, sheet_name="Fold distribution", index=False)
        results_df.to_excel(writer, sheet_name="Fold results", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        article_table_final.to_excel(writer, sheet_name="Article table", index=False)
        accumulated_df.to_excel(writer, sheet_name="Accumulated metrics", index=False)
except Exception as e:
    print(f"Não foi possível gerar o Excel: {e}")

print("\n" + "=" * 80)
print("MODO COMPLETO FINALIZADO")
print("=" * 80)
print(f"Arquivos salvos em: {OUTPUT_DIR}")
print("Arquivos principais:")
print(f" - {dataset_index_path.name}")
print(f" - {group_summary_path.name}")
print(f" - {fold_assignments_path.name}")
print(" - results_by_model_and_fold_FULL.csv")
print(" - summary_mean_std_FULL.csv")
print(" - new_training_FULL_results.xlsx")
print("=" * 80)
