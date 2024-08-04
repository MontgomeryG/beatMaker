import librosa
import numpy as np

def load_and_extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)
    
    # Extract tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    # Extract MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Extract chroma feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Extract spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Aggregate features into a dictionary
    features = {
        'tempo': tempo,
        'beat_frames': beat_frames,
        'mfcc': mfcc,
        'chroma': chroma,
        'spectral_contrast': spectral_contrast
    }
    
    return features