# Hummingbird NAS + ANN Trainer (4 to 6 Layer FFNN Version)
# ---------------------------------------------------------
# Rewritten by Grimoire
# Fully functional, Windows-safe multiprocessing, TensorFlow-safe import pattern
# ---------------------------------------------------------

import os, random, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import multiprocessing as mp

# ----------------------------------------------------------------
# GLOBALS
# ----------------------------------------------------------------
CACHE_DIR = './cache'
OUTPUT_ROOT = './Hummingbird_NAS_FFNN_4to6'
os.makedirs(OUTPUT_ROOT, exist_ok=True)

seed = 42
np.random.seed(seed)
random.seed(seed)

# ----------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------
def load_npz(name, key='feats'):
    return np.load(os.path.join(CACHE_DIR, f"{name}.npz"))[key]

train_data = np.load(os.path.join(CACHE_DIR, 'train_images.npz'))
labels = train_data['labels'].astype(int).flatten()
features = load_npz('train_selected')
feature_dim = features.shape[1]

# ----------------------------------------------------------------
# ARCHITECTURE ENCODING (4 to 6 Layers)
# ----------------------------------------------------------------
static_layers = 6  # upper bound for gene length encoding
arch_gene_dim = 1 + static_layers * 2
full_gene_dim = feature_dim + arch_gene_dim


def decode_architecture(gene):
    mask = gene[:feature_dim] > 0.5
    arch = gene[feature_dim:]

    # 4 to 6 layers instead of 1–4
    L = int(4 + arch[0] * 2)
    units = []
    drops = []

    idx = 1
    for _ in range(L):
        u = int(32 + arch[idx] * 480)
        idx += 1
        d = arch[idx] * 0.5
        idx += 1
        units.append(max(8, u))
        drops.append(d)

    return mask, units, drops

# ----------------------------------------------------------------
# WORKER WRAPPER
# ----------------------------------------------------------------
def worker_eval(args):
    return evaluate_bird(*args)

# ----------------------------------------------------------------
# WORKER FUNCTION
# ----------------------------------------------------------------
def evaluate_bird(gene, exp_dir):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    mask, units, drops = decode_architecture(gene)

    if mask.sum() < 5:
        return -1e9, gene

    X = features[:, mask]
    y = labels

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    cw_vals = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    cw = {i: w for i, w in enumerate(cw_vals)}

    def build_ann():
        model = Sequential()
        model.add(Dense(units[0], activation='relu', input_dim=X.shape[1]))
        model.add(BatchNormalization())
        model.add(Dropout(drops[0]))

        for i in range(1, len(units)):
            model.add(Dense(units[i], activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(drops[i]))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    model = build_ann()

    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early, reduce],
        class_weight=cw,
        verbose=0
    )

    if max(hist.history['val_accuracy']) < 0.65:
        return -1e9, gene

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    acc = (y_pred == y_test).mean()
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0

    f1 = f1_score(y_test, y_pred)
    params = model.count_params()
    fitness = acc + auc + f1 - params * 1e-6

    os.makedirs(exp_dir, exist_ok=True)

    np.save(os.path.join(exp_dir, "gene_vector.npy"), gene)
    np.save(os.path.join(exp_dir, "selected_features.npy"), np.where(mask)[0])

    pd.DataFrame(hist.history).to_csv(os.path.join(exp_dir, "history.csv"), index=False)
    pd.DataFrame({"units": units, "drops": drops}).to_csv(os.path.join(exp_dir, "architecture.csv"), index=False)

    model.save(os.path.join(exp_dir, "SavedModel"))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(exp_dir, 'confusion_matrix.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='train acc')
    plt.plot(hist.history['val_accuracy'], label='val acc')
    plt.legend(); plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='train loss')
    plt.plot(hist.history['val_loss'], label='val loss')
    plt.legend(); plt.title('Loss')
    plt.savefig(os.path.join(exp_dir, 'training_curves.png'))
    plt.close()

    pd.DataFrame({
        "acc": [acc],
        "auc": [auc],
        "f1": [f1],
        "params": [params],
        "fitness": [fitness]
    }).to_csv(os.path.join(exp_dir, 'metrics.csv'), index=False)

    return fitness, gene

# ----------------------------------------------------------------
# HUMMINGBIRD SEARCH
# ----------------------------------------------------------------
def hummingbird_search(birds=6, gens=10):
    swarm = np.random.uniform(0, 1, (birds, full_gene_dim))
    fitness = np.zeros(birds)

    pool = mp.Pool(processes=max(1, mp.cpu_count() - 1))

    tasks = [(swarm[i].copy(), os.path.join(OUTPUT_ROOT, f"Experiment_Init_{i}")) for i in range(birds)]
    results = pool.map(worker_eval, tasks)

    for i, (fit, gene) in enumerate(results):
        fitness[i] = fit
        swarm[i] = gene

    best_idx = np.argmax(fitness)
    best_gene = swarm[best_idx]
    best_fit = fitness[best_idx]

    print(f"Initial best fitness: {best_fit:.4f}")

    for g in range(1, gens + 1):
        print(f"\nGeneration {g}/{gens}")
        new_swarm = swarm.copy()

        for i in range(birds):
            j, k = np.random.choice([x for x in range(birds) if x != i], 2, replace=False)
            new_gene = swarm[i] + np.random.uniform(-0.1, 0.1, full_gene_dim) * (swarm[j] - swarm[k])
            new_gene = np.clip(new_gene, 0, 1)
            new_swarm[i] = new_gene

        tasks = [(new_swarm[i].copy(), os.path.join(OUTPUT_ROOT, f"Experiment_G{g}_{i}")) for i in range(birds)]
        results = pool.map(worker_eval, tasks)

        for i, (fit, gene) in enumerate(results):
            if fit > fitness[i]:
                fitness[i] = fit
                swarm[i] = gene

        gen_best_idx = np.argmax(fitness)
        gen_best_fit = fitness[gen_best_idx]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_gene = swarm[gen_best_idx]

        print(f"Best fitness this generation: {gen_best_fit:.4f}, Overall best: {best_fit:.4f}")

    pool.close(); pool.join()

    np.save(os.path.join(OUTPUT_ROOT, 'best_gene.npy'), best_gene)
    print("\nSearch completed. Best model saved.")

if __name__ == '__main__':
    hummingbird_search(birds=4, gens=5)
