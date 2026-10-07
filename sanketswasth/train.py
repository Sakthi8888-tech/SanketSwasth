"""Train a small LSTM on collected landmark sequences and export to TFLite.

    python -m sanketswasth.train --data data --epochs 60
"""
import argparse
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load(data_dir):
    labels = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    X, y = [], []
    for i, lab in enumerate(labels):
        for f in os.listdir(os.path.join(data_dir, lab)):
            if f.endswith(".npy"):
                X.append(np.load(os.path.join(data_dir, lab, f)))
                y.append(i)
    return np.array(X), np.array(y), labels


def build(n_classes, seq_len, feats):
    return tf.keras.Sequential([
        tf.keras.layers.Input((seq_len, feats)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()

    X, y, labels = load(a.data)
    if len(X) == 0:
        raise SystemExit("No data found. Run sanketswasth.collect first.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = build(len(labels), X.shape[1], X.shape[2])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=a.epochs, batch_size=16,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)])

    pred = model.predict(Xte).argmax(1)
    print(classification_report(yte, pred, labels=range(len(labels)),
                                target_names=labels, zero_division=0))

    os.makedirs(a.out, exist_ok=True)
    model.save(os.path.join(a.out, "sign_model.keras"))
    json.dump(labels, open(os.path.join(a.out, "labels.json"), "w"))

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    conv._experimental_lower_tensor_list_ops = False
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    open(os.path.join(a.out, "sign_model.tflite"), "wb").write(conv.convert())
    print("Saved model + TFLite to", a.out)


if __name__ == "__main__":
    main()