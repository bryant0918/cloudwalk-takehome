import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import pyaudio
import numpy as np
import wave
import time
import os
from inference import DigitClassifierInference
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa

class DigitClassifierGUI:
    """GUI application for digit classification with microphone integration"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Classifier - Audio Recognition")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        # Audio recording parameters
        self.sample_rate = 8000
        self.chunk_size = 1024
        self.record_duration = 1.0  # 1 second
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Recording state
        self.is_recording = False
        self.recorded_audio = None
        self.recorded_audio_raw = None  # Store raw audio for playback
        self.audio_stream = None
        self.pyaudio_instance = None
        self.processing_time = 0  # Store processing time
        
        # Initialize classifier
        self.classifier = None
        self.init_classifier()
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize matplotlib figure
        self.setup_plot()
        
    def init_classifier(self):
        """Initialize the digit classifier"""
        try:
            # Try to load the best model first, fallback to final model
            try:
                self.classifier = DigitClassifierInference('best_digit_classifier_enhanced.pth')
                self.model_status = "Model loaded: best_digit_classifier_enhanced.pth"
            except:
                self.classifier = DigitClassifierInference('digit_classifier_enhanced_final.pth')
                self.model_status = "Model loaded: digit_classifier_enhanced_final.pth"
        except Exception as e:
            self.classifier = None
            self.model_status = f"Error loading model: {str(e)}"
            messagebox.showerror("Model Error", 
                               f"Failed to load model: {str(e)}\n\n"
                               "Make sure you have trained the model first!")
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Digit Classifier", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Model status
        self.status_label = ttk.Label(main_frame, text=self.model_status, 
                                     font=("Arial", 10))
        self.status_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=3, pady=(0, 20))
        
        # Record button
        self.record_button = ttk.Button(control_frame, text="🎤 Record (1s)", 
                                       command=self.toggle_recording,
                                       style="Accent.TButton")
        self.record_button.grid(row=0, column=0, padx=(0, 10))
        
        # Load file button
        self.load_button = ttk.Button(control_frame, text="📁 Load Audio File", 
                                     command=self.load_audio_file)
        self.load_button.grid(row=0, column=1, padx=(0, 10))
        
        # Clear button
        self.clear_button = ttk.Button(control_frame, text="🗑️ Clear", 
                                      command=self.clear_results)
        self.clear_button.grid(row=0, column=2)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), 
                          pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Prediction display
        pred_frame = ttk.Frame(results_frame)
        pred_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        pred_frame.columnconfigure(1, weight=1)
        
        ttk.Label(pred_frame, text="Predicted Digit:", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky=tk.W)
        self.prediction_label = ttk.Label(pred_frame, text="--", 
                                         font=("Arial", 24, "bold"), 
                                         foreground="blue")
        self.prediction_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(pred_frame, text="Confidence:", font=("Arial", 12, "bold")).grid(
            row=1, column=0, sticky=tk.W)
        self.confidence_label = ttk.Label(pred_frame, text="--", 
                                         font=("Arial", 12))
        self.confidence_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(pred_frame, text="Response Time:", font=("Arial", 12, "bold")).grid(
            row=2, column=0, sticky=tk.W)
        self.response_time_label = ttk.Label(pred_frame, text="--", 
                                           font=("Arial", 12))
        self.response_time_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # Playback button
        self.playback_button = ttk.Button(pred_frame, text="🔊 Play Audio", 
                                         command=self.play_audio, state="disabled")
        self.playback_button.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        
        # Plot frame for audio waveform and probabilities
        self.plot_frame = ttk.Frame(results_frame)
        self.plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), 
                            pady=(10, 0))
    
    def setup_plot(self):
        """Setup matplotlib plots"""
        # Create figure with better proportions and spacing
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)   

        # Audio waveform plot
        self.ax1.set_title("Audio Waveform", fontsize=10, fontweight='bold')
        self.ax1.set_xlabel("Time (s)", fontsize=8)
        self.ax1.set_ylabel("Amplitude", fontsize=8)
        self.ax1.grid(True, alpha=0.3)
        
        # Probability distribution plot
        self.ax2.set_title("Prediction Probabilities", fontsize=10, fontweight='bold')
        self.ax2.set_xlabel("Digit", fontsize=8)
        self.ax2.set_ylabel("Probability", fontsize=8)
        self.ax2.set_xticks(range(10))
        self.ax2.grid(True, alpha=0.3)
        
        # Embed plot in tkinter with better configuration
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.canvas.get_tk_widget().configure(height=600)  # Set minimum height
        
        # Clear initial plots
        self.clear_plots()
    
    def clear_plots(self):
        """Clear the plots"""
        self.ax1.clear()
        self.ax1.set_title("Audio Waveform")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.clear()
        self.ax2.set_title("Prediction Probabilities")
        self.ax2.set_xlabel("Digit")
        self.ax2.set_ylabel("Probability")
        self.ax2.set_xticks(range(10))
        self.ax2.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def toggle_recording(self):
        """Toggle audio recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording"""
        if self.classifier is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        try:
            self.is_recording = True
            self.record_button.config(text="⏹️ Stop Recording", style="")
            self.update_status("Recording...")
            
            # Start recording in a separate thread
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
        except Exception as e:
            self.is_recording = False
            self.record_button.config(text="🎤 Record (1s)", style="Accent.TButton")
            messagebox.showerror("Recording Error", f"Failed to start recording: {str(e)}")
            self.update_status("Ready")
    
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        self.record_button.config(text="🎤 Record (1s)", style="Accent.TButton")
    
    def _record_audio(self):
        """Record audio in a separate thread"""
        try:
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Record audio
            frames = []
            num_chunks = int(self.sample_rate * self.record_duration / self.chunk_size)
            
            for i in range(num_chunks):
                if not self.is_recording:
                    break
                data = self.audio_stream.read(self.chunk_size)
                frames.append(data)
            
            # Close stream
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.pyaudio_instance.terminate()
            
            # Convert to numpy array
            audio_data = b''.join(frames)
            self.recorded_audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            self.recorded_audio = self.recorded_audio / 32768.0  # Normalize to [-1, 1]
            
            # Process the recorded audio
            self.root.after(0, self._process_recorded_audio)
            
        except Exception as e:
            self.root.after(0, lambda: self._handle_recording_error(str(e)))
    
    def _handle_recording_error(self, error_msg):
        """Handle recording errors in main thread"""
        self.is_recording = False
        self.record_button.config(text="🎤 Record (1s)", style="Accent.TButton")
        messagebox.showerror("Recording Error", f"Recording failed: {error_msg}")
        self.update_status("Ready")
    
    def _process_recorded_audio(self):
        """Process recorded audio in main thread"""
        try:
            self.update_status("Processing audio...")
            
            # Measure processing time
            start_time = time.time()
            
            # Make prediction
            predicted_digit, confidence, probabilities = self.classifier.predict(
                self.recorded_audio, self.sample_rate, return_confidence=True
            )
            
            # Calculate processing time
            self.processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Store raw audio for playback (convert back to int16)
            self.recorded_audio_raw = (self.recorded_audio * 32768.0).astype(np.int16)
            
            # Update GUI
            self.update_results(predicted_digit, confidence, probabilities, 
                              self.recorded_audio, self.sample_rate)
            
            self.update_status("Prediction complete")
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"Failed to process audio: {str(e)}")
            self.update_status("Ready")
        finally:
            self.is_recording = False
            self.record_button.config(text="🎤 Record (1s)", style="Accent.TButton")
    
    def load_audio_file(self):
        """Load audio file for prediction"""
        if self.classifier is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.update_status("Loading audio file...")
                
                # Load audio file
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                
                # Measure processing time
                start_time = time.time()
                
                # Make prediction
                predicted_digit, confidence, probabilities = self.classifier.predict(
                    audio_data, sample_rate, return_confidence=True
                )
                
                # Calculate processing time
                self.processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Store raw audio for playback (convert to int16 if needed)
                if audio_data.dtype != np.int16:
                    self.recorded_audio_raw = (audio_data * 32768.0).astype(np.int16)
                else:
                    self.recorded_audio_raw = audio_data
                
                # Update GUI
                self.update_results(predicted_digit, confidence, probabilities, 
                                  audio_data, sample_rate)
                
                self.update_status(f"Loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("File Error", f"Failed to load audio file: {str(e)}")
                self.update_status("Ready")
    
    def update_results(self, predicted_digit, confidence, probabilities, audio_data, sample_rate):
        """Update the results display"""
        # Update prediction labels
        self.prediction_label.config(text=str(predicted_digit))
        self.confidence_label.config(text=f"{confidence:.2%}")
        self.response_time_label.config(text=f"{self.processing_time:.1f} ms")
        
        # Enable playback button if we have audio
        if self.recorded_audio_raw is not None:
            self.playback_button.config(state="normal")
        
        # Update plots
        self.update_plots(audio_data, sample_rate, probabilities)
    
    def update_plots(self, audio_data, sample_rate, probabilities):
        """Update the audio waveform and probability plots"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot audio waveform
        time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        self.ax1.plot(time_axis, audio_data, 'b-', linewidth=1)
        self.ax1.set_title("Audio Waveform")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, alpha=0.3)
        
        # Plot probability distribution
        digits = range(10)
        bars = self.ax2.bar(digits, probabilities, color='skyblue', alpha=0.7)
        
        # Highlight the predicted digit
        max_idx = np.argmax(probabilities)
        bars[max_idx].set_color('red')
        bars[max_idx].set_alpha(1.0)
        
        self.ax2.set_title("Prediction Probabilities")
        self.ax2.set_xlabel("Digit")
        self.ax2.set_ylabel("Probability")
        self.ax2.set_xticks(digits)
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        
        # Add probability values on bars
        for i, (digit, prob) in enumerate(zip(digits, probabilities)):
            self.ax2.text(digit, prob + 0.01, f'{prob:.3f}', 
                         ha='center', va='bottom', fontsize=8)
        
        # Refresh canvas
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
    
    def play_audio(self):
        """Play the recorded audio"""
        if self.recorded_audio_raw is None:
            messagebox.showwarning("No Audio", "No audio to play!")
            return
        
        try:
            self.update_status("Playing audio...")
            
            # Start playback in a separate thread
            playback_thread = threading.Thread(target=self._play_audio_thread)
            playback_thread.daemon = True
            playback_thread.start()
            
        except Exception as e:
            messagebox.showerror("Playback Error", f"Failed to play audio: {str(e)}")
            self.update_status("Ready")
    
    def _play_audio_thread(self):
        """Play audio in a separate thread"""
        try:
            # Initialize PyAudio for playback
            p = pyaudio.PyAudio()
            
            # Open audio stream for playback
            stream = p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Convert numpy array to bytes
            audio_bytes = self.recorded_audio_raw.tobytes()
            
            # Play audio in chunks
            chunk_size_bytes = self.chunk_size * 2  # 2 bytes per sample for int16
            for i in range(0, len(audio_bytes), chunk_size_bytes):
                chunk = audio_bytes[i:i + chunk_size_bytes]
                stream.write(chunk)
            
            # Close stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Update status in main thread
            self.root.after(0, lambda: self.update_status("Playback complete"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Playback Error", f"Audio playback failed: {str(e)}"))
            self.root.after(0, lambda: self.update_status("Ready"))
    
    def clear_results(self):
        """Clear all results"""
        self.prediction_label.config(text="--")
        self.confidence_label.config(text="--")
        self.response_time_label.config(text="--")
        self.playback_button.config(state="disabled")
        self.clear_plots()
        self.recorded_audio = None
        self.recorded_audio_raw = None
        self.processing_time = 0
        self.update_status("Ready")
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_recording:
            self.stop_recording()
            time.sleep(0.1)  # Give time for recording to stop
        
        if hasattr(self, 'audio_stream') and self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
        
        if hasattr(self, 'pyaudio_instance') and self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass
        
        self.root.destroy()

def main():
    """Main function to run the GUI application"""
    # Check if required files exist
    model_files = ['best_digit_classifier_enhanced.pth', 'digit_classifier_enhanced_final.pth']
    if not any(os.path.exists(f) for f in model_files):
        print("Error: No trained model found!")
        print("Please make sure you have either 'best_digit_classifier_enhanced.pth' or")
        print("'digit_classifier_enhanced_final.pth' in the current directory.")
        print("Run the training script first: python digit_classifier_enhanced.py")
        return
    
    # Create and run GUI
    root = tk.Tk()
    app = DigitClassifierGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("Starting Digit Classifier GUI...")
    print("Make sure your microphone is connected and working.")
    print("You can record 1-second audio clips or load audio files for prediction.")
    
    root.mainloop()

if __name__ == "__main__":
    main()
