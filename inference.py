import torch
import torch.nn as nn
import librosa
import numpy as np
from digit_classifier_enhanced import EnhancedDigitClassifier, AttentionPooling

class DigitClassifierInference:
    """Inference class for the enhanced digit classifier"""
    
    def __init__(self, model_path='best_digit_classifier_enhanced.pth', device=None):
        """
        Initialize the inference class
        
        Args:
            model_path (str): Path to the trained model checkpoint
            device (str): Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.load_model()
        
    def _get_device(self, device=None):
        """Get the best available device"""
        if device is not None:
            return torch.device(device)
            
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
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Initialize model architecture
            self.model = EnhancedDigitClassifier(num_classes=10, input_length=8000, dropout_rate=0.3)
            
            # Load trained weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Running on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_audio(self, audio_data, sample_rate=None, target_length=8000):
        """
        Preprocess audio data for inference
        
        Args:
            audio_data (np.array): Raw audio data
            sample_rate (int): Sample rate of the audio data
            target_length (int): Target length for the audio (default: 8000 for 1 second at 8kHz)
            
        Returns:
            torch.Tensor: Preprocessed audio tensor
        """
        # Convert to numpy array if needed
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)
        elif not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        # Resample to 8kHz if needed
        if sample_rate is not None and sample_rate != 8000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=8000)
        
        # Pad or truncate to target length
        if len(audio_data) > target_length:
            # Center crop
            start_idx = (len(audio_data) - target_length) // 2
            audio_data = audio_data[start_idx:start_idx + target_length]
        else:
            # Pad with zeros
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
        
        # Normalize audio
        if np.std(audio_data) > 0:
            audio_data = (audio_data - np.mean(audio_data)) / np.std(audio_data)
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        
        return audio_tensor
    
    def predict(self, audio_data, sample_rate=None, return_confidence=False):
        """
        Predict digit from audio data
        
        Args:
            audio_data (np.array): Raw audio data
            sample_rate (int): Sample rate of the audio data
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            int or tuple: Predicted digit (and confidence if requested)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess audio
        audio_tensor = self.preprocess_audio(audio_data, sample_rate)
        audio_tensor = audio_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_digit = torch.argmax(outputs, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
        
        if return_confidence:
            return predicted_digit, confidence, probabilities.cpu().numpy()[0]
        else:
            return predicted_digit
    
    def predict_from_file(self, audio_file_path, return_confidence=False):
        """
        Predict digit from audio file
        
        Args:
            audio_file_path (str): Path to audio file
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            int or tuple: Predicted digit (and confidence if requested)
        """
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
            
            # Make prediction
            return self.predict(audio_data, sample_rate, return_confidence)
            
        except Exception as e:
            print(f"Error processing audio file {audio_file_path}: {e}")
            raise
    
    def batch_predict(self, audio_list, sample_rates=None, return_confidence=False):
        """
        Predict digits from a batch of audio data
        
        Args:
            audio_list (list): List of audio arrays
            sample_rates (list): List of sample rates (optional)
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            list: List of predictions (and confidences if requested)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        
        for i, audio_data in enumerate(audio_list):
            sample_rate = sample_rates[i] if sample_rates else None
            result = self.predict(audio_data, sample_rate, return_confidence)
            results.append(result)
        
        return results

def demo_inference():
    """Demo function to test inference"""
    print("Initializing digit classifier inference...")
    
    # Try to load the best model first, fallback to final model
    try:
        classifier = DigitClassifierInference('best_digit_classifier_enhanced.pth')
    except:
        try:
            classifier = DigitClassifierInference('digit_classifier_enhanced_final.pth')
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have trained the model first!")
            return
    
    # Generate a simple test audio (sine wave)
    print("\nGenerating test audio...")
    duration = 1.0  # 1 second
    sample_rate = 8000
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Make prediction
    print("Making prediction...")
    predicted_digit, confidence, probabilities = classifier.predict(
        test_audio, sample_rate, return_confidence=True
    )
    
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.4f}")
    print("All probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  Digit {i}: {prob:.4f}")

if __name__ == "__main__":
    demo_inference()
