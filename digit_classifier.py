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
warnings.filterwarnings('ignore')

class DigitAudioDataset(Dataset):
    """Custom dataset for digit audio classification"""
    
    def __init__(self, dataset, processor, max_length=16000):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get audio array and sampling rate
        audio = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        
        # Resample to 16kHz if needed (Wav2Vec2 expects 16kHz)
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        
        # Pad or truncate to fixed length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
        
        # Process audio with Wav2Vec2 processor
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Get label (digit)
        label = item['label']
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class DigitClassifier(nn.Module):
    """Digit classifier using pre-trained Wav2Vec2 + classification head"""
    
    def __init__(self, num_classes=10, freeze_feature_extractor=True):
        super().__init__()
        
        # Load pre-trained Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze the feature extractor if specified
        if freeze_feature_extractor:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.wav2vec2.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_values):
        # Extract features using Wav2Vec2
        outputs = self.wav2vec2(input_values)
        
        # Use mean pooling over the sequence dimension
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3):
    """Train the digit classifier"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_accuracies = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for batch in train_pbar:
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')
            for batch in val_pbar:
                input_values = batch['input_values'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_values)
                predictions = torch.argmax(outputs, dim=1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.4f}')
        print('-' * 50)
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader):
    """Evaluate the model and return detailed metrics"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    labels = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for batch in test_pbar:
            input_values = batch['input_values'].to(device)
            batch_labels = batch['label'].to(device)
            
            outputs = model(input_values)
            batch_predictions = torch.argmax(outputs, dim=1)
            
            predictions.extend(batch_predictions.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=[str(i) for i in range(10)])
    cm = confusion_matrix(labels, predictions)
    
    return accuracy, report, cm, predictions, labels

def plot_results(train_losses, val_accuracies, confusion_matrix):
    """Plot training results and confusion matrix"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Training loss
    axes[0].plot(train_losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Validation accuracy
    axes[1].plot(val_accuracies)
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)
    
    # Confusion matrix
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title('Confusion Matrix')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def digit_classifier(audio_path):
    """
    Main inference function for digit classification
    
    Args:
        audio_path: Path to audio file or numpy array of audio data
    
    Returns:
        predicted_digit: Integer from 0-9
        confidence: Confidence score
    """
    
    # Load the trained model (this would be loaded from saved checkpoint in practice)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For demo purposes, we'll create a dummy model
    # In practice, you'd load: model = torch.load('digit_classifier_model.pth')
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = DigitClassifier()
    model.to(device)
    model.eval()
    
    # Load and preprocess audio
    if isinstance(audio_path, str):
        audio, sr = librosa.load(audio_path, sr=16000)
    else:
        audio = audio_path
        sr = 16000
    
    # Ensure fixed length
    max_length = 16000  # 1 second at 16kHz
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
    
    # Process with Wav2Vec2 processor
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_values)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_digit = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_digit].item()
    
    return predicted_digit, confidence

def main():
    """Main training and evaluation pipeline"""
    
    print("Loading Free Spoken Digit Dataset...")
    
    # Load the dataset
    dataset = load_dataset("mteb/free-spoken-digit-dataset")
    
    print(f"Dataset loaded successfully!")
    print(f"Train set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")
    
    # Initialize processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Create datasets
    train_dataset = DigitAudioDataset(dataset['train'], processor)
    test_dataset = DigitAudioDataset(dataset['test'], processor)
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create validation split from training data
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = DigitClassifier(num_classes=10, freeze_feature_extractor=True)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=5, learning_rate=1e-3
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_report, test_cm, predictions, labels = evaluate_model(model, test_loader)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(test_report)
    
    # Plot results
    plot_results(train_losses, val_accuracies, test_cm)
    
    # Save model
    torch.save(model.state_dict(), 'digit_classifier_model.pth')
    print("\nModel saved as 'digit_classifier_model.pth'")
    
    # Demo inference
    print("\nDemo: Testing inference on a sample...")
    sample_audio = dataset['test'][0]['audio']['array']
    predicted_digit, confidence = digit_classifier(sample_audio)
    actual_digit = dataset['test'][0]['label']
    
    print(f"Actual digit: {actual_digit}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.4f}")
    
    return model, test_accuracy

if __name__ == "__main__":
    model, accuracy = main()
