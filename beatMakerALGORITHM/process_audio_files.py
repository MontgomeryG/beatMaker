import os
from load_and_extract_features import load_and_extract_features

def process_audio_files(directory):
    feature_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            file_path = os.path.join(directory, filename)
            features = load_and_extract_features(file_path)
            feature_list.append(features)
    return feature_list
