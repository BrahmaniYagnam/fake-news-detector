from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

from backend.config.settings import settings
from backend.models.bert_classifier import BertClassifier
from training.data import FakeNewsDataset, load_dataset, prepare_data_loaders, save_labels, save_metadata


def train_model(dataset_path: Path) -> None:
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataframe = load_dataset(dataset_path)
    loaders = prepare_data_loaders(dataframe, tokenizer)
    train_loader = loaders["train"]
    val_loader = loaders["validation"]

    model = BertClassifier(dropout=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    metrics: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for epoch in range(1, 21):
        model.train()
        epoch_losses: list[float] = []
        epoch_preds: list[int] = []
        epoch_labels: list[int] = []

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            epoch_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            epoch_labels.extend(labels.cpu().tolist())

        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_acc = accuracy_score(epoch_labels, epoch_preds)
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_acc)

        model.eval()
        val_losses: list[float] = []
        val_preds: list[int] = []
        val_labels: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
                val_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        val_acc = accuracy_score(val_labels, val_preds) if val_labels else 0
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if val_loss <= best_val_loss or epoch == 20:
            best_val_loss = val_loss
            model_path = settings.model_dir / "pytorch_model.bin"
            settings.model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)

    print("Training finished.")

    save_labels(settings.model_dir, loaders["label_map"])
    save_metadata(
        settings.model_dir,
        {
            "model_name": settings.model_name,
            "version": settings.app_version,
            "framework": "PyTorch",
            "dataset": "Fake and Real News Dataset",
            "training_accuracy": metrics["train_accuracy"][-1],
        },
    )
    metrics_path = settings.model_dir / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fake news detection model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/fake_or_real_news.csv",
        help="Path to the CSV dataset file",
    )
    args = parser.parse_args()
    train_model(Path(args.dataset))
