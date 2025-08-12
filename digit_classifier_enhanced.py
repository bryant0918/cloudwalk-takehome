import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
warnings.filterwarnings('ignore')

class AudioAugmentation:
    """Audio data augmentation techniques"""
    
    def __init__(self, sample_rate=8000):
        self.sample_rate = sample_rate
    
    def add_noise(self, audio, noise_factor=0.005):
        """Add random noise to audio"""
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    def time_stretch(self, audio, rate=None):
        """Time stretching without changing pitch"""
        if rate is None:
            rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, n_steps=None):
        """Pitch shifting without changing tempo"""
        if n_steps is None:
            n_steps = np.random.uniform(-2, 2)
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def time_mask(self, audio, max_mask_pct=0.1):
        """Mask random time segments"""
        mask_length = int(len(audio) * max_mask_pct * np.random.random())
        mask_start = np.random.randint(0, max(1, len(audio) - mask_length))
        audio_masked = audio.copy()
        audio_masked[mask_start:mask_start + mask_length] = 0
        return audio_masked
    
    def volume_change(self, audio, factor=None):
        """Change volume by random factor"""
        if factor is None:
            factor = np.random.uniform(0.7, 1.3)
        return audio * factor

class EnhancedDigitAudioDataset(Dataset):
    """Enhanced dataset with data augmentation and better preprocessing"""
    
    def __init__(self, dataset, processor, max_length=8000, augment=False, augment_prob=0.5):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length  # Keep original 8kHz length
        self.augment = augment
        self.augment_prob = augment_prob
        self.augmentation = AudioAugmentation(sample_rate=8000)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get audio array and sampling rate
        audio = item['audio']['array'].astype(np.float32)
        sampling_rate = item['audio']['sampling_rate']
        
        # Keep original 8kHz sampling rate (no need to resample to 16kHz)
        if sampling_rate != 8000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=8000)
        
        # Apply augmentation during training
        if self.augment and random.random() < self.augment_prob:
            aug_choice = random.choice(['noise', 'time_stretch', 'pitch_shift', 'time_mask', 'volume'])
            
            if aug_choice == 'noise':
                audio = self.augmentation.add_noise(audio)
            elif aug_choice == 'time_stretch':
                try:
                    audio = self.augmentation.time_stretch(audio)
                except:
                    pass  # Skip if augmentation fails
            elif aug_choice == 'pitch_shift':
                try:
                    audio = self.augmentation.pitch_shift(audio)
                except:
                    pass
            elif aug_choice == 'time_mask':
                audio = self.augmentation.time_mask(audio)
            elif aug_choice == 'volume':
                audio = self.augmentation.volume_change(audio)
        
        # Pad or truncate to fixed length
        if len(audio) > self.max_length:
            # Random crop during training, center crop during testing
            if self.augment:
                start_idx = random.randint(0, len(audio) - self.max_length)
                audio = audio[start_idx:start_idx + self.max_length]
            else:
                # Center crop
                start_idx = (len(audio) - self.max_length) // 2
                audio = audio[start_idx:start_idx + self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
        
        # Normalize audio
        if np.std(audio) > 0:
            audio = (audio - np.mean(audio)) / np.std(audio)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        # Get label (digit)
        label = item['label']
        
        return {
            'audio': audio_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }

class AttentionPooling(nn.Module):
    """Attention-based pooling layer"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        pooled = torch.sum(attention_weights * x, dim=1)
        return pooled

class EnhancedDigitClassifier(nn.Module):
    """Enhanced digit classifier with better architecture"""
    
    def __init__(self, num_classes=10, input_length=8000, dropout_rate=0.3):
        super().__init__()
        
        # CNN feature extractor optimized for 8kHz audio
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38),  # ~20ms windows at 8kHz
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout_rate * 0.5),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout_rate * 0.5),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout_rate * 0.5),
            
            # Fourth conv block
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Use regular pooling instead of adaptive
        )
        
        # Attention pooling
        self.attention_pool = AttentionPooling(512)
        
        # Classification head with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, seq_len)
        
        # CNN feature extraction
        x = self.conv_layers(x)  # (batch_size, 512, 32)
        
        # Transpose for attention
        x = x.transpose(1, 2)  # (batch_size, 32, 512)
        
        # Attention pooling
        x = self.attention_pool(x)  # (batch_size, 512)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_device():
    """Get the best available device (MPS for Mac, CUDA for NVIDIA, CPU otherwise)"""
    try:
        if torch.backends.mps.is_available():
            # Test MPS with a simple operation
            test_tensor = torch.randn(10, 10, device='mps')
            _ = test_tensor @ test_tensor
            return torch.device('mps')
    except Exception as e:
        print(f"MPS device available but has issues: {e}")
        print("Falling back to CPU...")
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train_enhanced_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-3, 
                        use_focal_loss=False, patience=5):
    """Enhanced training with better optimization and scheduling"""
    
    device = get_device()
    model.to(device)
    
    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience_counter = 0
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for batch in train_pbar:
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct_train / total_train:.2f}%'
            })
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')
            for batch in val_pbar:
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(audio)
                predictions = torch.argmax(outputs, dim=1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_digit_classifier_enhanced.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
            
        print('-' * 60)
    
    # Load best model
    model.load_state_dict(torch.load('best_digit_classifier_enhanced.pth'))
    
    return train_losses, val_accuracies, best_val_acc

def evaluate_enhanced_model(model, test_loader):
    """Enhanced evaluation with detailed metrics"""
    
    device = get_device()
    model.to(device)
    model.eval()
    
    predictions = []
    labels = []
    confidences = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for batch in test_pbar:
            audio = batch['audio'].to(device)
            batch_labels = batch['label'].to(device)
            
            outputs = model(audio)
            probabilities = torch.softmax(outputs, dim=1)
            batch_predictions = torch.argmax(outputs, dim=1)
            
            predictions.extend(batch_predictions.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            confidences.extend(torch.max(probabilities, dim=1)[0].cpu().numpy())
    
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=[str(i) for i in range(10)])
    cm = confusion_matrix(labels, predictions)
    
    return accuracy, report, cm, predictions, labels, confidences

def plot_enhanced_results(train_losses, val_accuracies, confusion_matrix, save_path='enhanced_training_results.png'):
    """Enhanced plotting with more detailed visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training loss
    axes[0, 0].plot(train_losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[0, 1].plot(val_accuracies, 'g-', linewidth=2)
    axes[0, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Confusion matrix
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Per-class accuracy
    per_class_acc = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    axes[1, 1].bar(range(10), per_class_acc, color='skyblue', alpha=0.7)
    axes[1, 1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Digit')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_xticks(range(10))
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Enhanced main training and evaluation pipeline"""
    
    print("Loading Free Spoken Digit Dataset...")
    
    # Load the dataset
    dataset = load_dataset("mteb/free-spoken-digit-dataset")
    
    print(f"Dataset loaded successfully!")
    print(f"Train set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")
    
    # Create enhanced datasets with augmentation
    train_dataset = EnhancedDigitAudioDataset(
        dataset['train'], 
        processor=None,  # We don't need Wav2Vec2 processor anymore
        max_length=8000,  # 1 second at 8kHz
        augment=True,
        augment_prob=0.7
    )
    
    test_dataset = EnhancedDigitAudioDataset(
        dataset['test'], 
        processor=None,
        max_length=8000,
        augment=False
    )
    
    # Create validation split from training data
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders with optimal batch size
    batch_size = 32
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize enhanced model
    model = EnhancedDigitClassifier(num_classes=10, input_length=8000, dropout_rate=0.3)
    
    print(f"Enhanced model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Train model with enhanced training loop
    print("\nStarting enhanced training...")
    train_losses, val_accuracies, best_val_acc = train_enhanced_model(
        model, train_loader, val_loader, 
        num_epochs=30, 
        learning_rate=1e-3,
        use_focal_loss=False,
        patience=7
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_report, test_cm, predictions, labels, confidences = evaluate_enhanced_model(model, test_loader)
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Improvement over baseline: {((test_accuracy - 0.36) / 0.36 * 100):.1f}%")
    print("\nClassification Report:")
    print(test_report)
    
    # Plot enhanced results
    plot_enhanced_results(train_losses, val_accuracies, test_cm)
    
    # Save final model
    torch.save(model.state_dict(), 'digit_classifier_enhanced_final.pth')
    print("\nEnhanced model saved as 'digit_classifier_enhanced_final.pth'")
    
    # Analyze confidence scores
    avg_confidence = np.mean(confidences)
    print(f"\nAverage prediction confidence: {avg_confidence:.4f}")
    
    return model, test_accuracy, best_val_acc

if __name__ == "__main__":
    model, test_accuracy, val_accuracy = main()
