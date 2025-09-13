#!/usr/bin/env python3
"""
Model Download Script for DriveSafe Drowsiness Detection System

This script helps download the pre-trained model for yawning detection.
If the Google Drive link is not accessible, it provides alternative solutions.
"""

import os
import requests
import gdown
from pathlib import Path

def download_model():
    """Download the ResNet50V2 model for yawning detection"""
    
    # Configuration
    MODEL_DIR = Path("model")
    MODEL_FILE_ID = "1UInMiIbaHChmI-KSQ7VRMp_53RZpSDd4"
    MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    MODEL_PATH = MODEL_DIR / "resnet50v2_model.keras"
    
    # Create model directory
    MODEL_DIR.mkdir(exist_ok=True)
    
    print("🚀 DriveSafe Model Downloader")
    print("=" * 40)
    
    # Check if model already exists
    if MODEL_PATH.exists():
        print(f"✅ Model already exists at: {MODEL_PATH}")
        response = input("🔄 Do you want to re-download? (y/N): ").lower()
        if response != 'y':
            print("📁 Using existing model.")
            return True
    
    print(f"📥 Attempting to download model from Google Drive...")
    print(f"🔗 URL: {MODEL_URL}")
    
    try:
        # Try gdown first
        print("🔄 Trying gdown method...")
        gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)
        
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000:  # Check if file is not empty
            print(f"✅ Model downloaded successfully!")
            print(f"📁 Location: {MODEL_PATH}")
            print(f"📊 Size: {MODEL_PATH.stat().st_size / (1024*1024):.1f} MB")
            return True
        else:
            print("❌ Downloaded file appears to be empty or corrupted")
            MODEL_PATH.unlink(missing_ok=True)
            
    except Exception as e:
        print(f"❌ gdown failed: {str(e)}")
    
    # Try requests method
    try:
        print("🔄 Trying requests method...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000:
            print(f"✅ Model downloaded successfully!")
            print(f"📁 Location: {MODEL_PATH}")
            print(f"📊 Size: {MODEL_PATH.stat().st_size / (1024*1024):.1f} MB")
            return True
        else:
            print("❌ Downloaded file appears to be empty or corrupted")
            MODEL_PATH.unlink(missing_ok=True)
            
    except Exception as e:
        print(f"❌ requests failed: {str(e)}")
    
    # Provide manual download instructions
    print("\n" + "=" * 50)
    print("❌ Automatic download failed!")
    print("📋 Manual Download Instructions:")
    print("=" * 50)
    print(f"1. Open this URL in your browser:")
    print(f"   {MODEL_URL}")
    print(f"2. Download the file")
    print(f"3. Rename it to: resnet50v2_model.keras")
    print(f"4. Place it in the 'model' folder:")
    print(f"   {MODEL_PATH}")
    print("\n💡 Alternative Solutions:")
    print("   - Use a different Google Drive file ID")
    print("   - Train your own model")
    print("   - Use the system without yawning detection (eye detection still works)")
    
    return False

def create_sample_model():
    """Create a simple placeholder model for testing"""
    try:
        import tensorflow as tf
        
        print("\n🔧 Creating a simple placeholder model...")
        
        # Create a simple model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Save the model
        MODEL_DIR = Path("model")
        MODEL_DIR.mkdir(exist_ok=True)
        MODEL_PATH = MODEL_DIR / "resnet50v2_model.keras"
        
        model.save(MODEL_PATH)
        
        print(f"✅ Placeholder model created at: {MODEL_PATH}")
        print("⚠️  This is a basic model - for best results, use the trained ResNet50V2 model")
        
        return True
        
    except ImportError:
        print("❌ TensorFlow not available for creating placeholder model")
        return False
    except Exception as e:
        print(f"❌ Error creating placeholder model: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚗 DriveSafe Model Setup")
    print("=" * 30)
    
    # Try to download the model
    success = download_model()
    
    if not success:
        print("\n🤔 Would you like to create a placeholder model for testing?")
        response = input("(This will allow the app to run, but yawning detection won't be accurate) (y/N): ").lower()
        
        if response == 'y':
            create_sample_model()
        else:
            print("📝 You can run the app without the model - eye detection will still work!")
    
    print("\n🎉 Setup complete! Run 'python app.py' to start the application.")
