import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from config.config import Config


def train(model, train_dataset, val_dataset, tokenizer, model_type='multimodal'):
    device = Config.DEVICE
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    total_steps = len(train_loader) * Config.MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 3
    trigger_times = 0

    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.MAX_EPOCHS} - Training"):
            optimizer.zero_grad()

            input_ids = batch.get('input_ids', None)
            attention_mask = batch.get('attention_mask', None)
            images = batch.get('image', None)
            labels = batch['labels'].to(device)

            if input_ids is not None:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

            if images is not None:
                images = images.to(device)

            if model_type == 'multimodal':
                outputs = model(input_ids, attention_mask, images)
            elif model_type == 'text_only':
                outputs = model(input_ids, attention_mask)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.MAX_EPOCHS} - Validation"):
                input_ids = batch.get('input_ids', None)
                attention_mask = batch.get('attention_mask', None)
                images = batch.get('image', None)
                labels = batch['labels'].to(device)

                if input_ids is not None:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                if images is not None:
                    images = images.to(device)

                if model_type == 'multimodal':
                    outputs = model(input_ids, attention_mask, images)
                elif model_type == 'text_only':
                    outputs = model(input_ids, attention_mask)
                else:
                    outputs = model(images)

                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_PATH, f"{model_type}_model.pth"))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break
