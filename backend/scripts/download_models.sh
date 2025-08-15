#!/bin/bash
"""
Download script for required models and dependencies.
This script provides instructions for downloading necessary model files.
"""

echo "🌵 Sahara Model Download Instructions"
echo "======================================"
echo

echo "Required Models for Full Functionality:"
echo

echo "1. OpenCV Face Detection Models:"
echo "   - opencv_face_detector.pbtxt"
echo "   - opencv_face_detector_uint8.pb"
echo "   Download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"
echo "   Place in: ./models/emotion_models/"
echo

echo "2. FER2013 Emotion Recognition Model:"
echo "   - fer2013_mini_XCEPTION.102-0.66.hdf5"
echo "   This model needs to be trained or obtained from a compatible source"
echo "   Place in: ./models/emotion_models/"
echo

echo "3. Sentence Transformer Model:"
echo "   - all-MiniLM-L6-v2 (automatically downloaded via sentence-transformers)"
echo "   This will be downloaded automatically when first used"
echo

echo "4. NVIDIA Riva Models:"
echo "   - Requires NVIDIA Riva server setup"
echo "   - See: https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html"
echo "   - Configure RIVA_URL and RIVA_API_KEY in .env file"
echo

echo "Directory Structure:"
echo "models/"
echo "├── emotion_models/"
echo "│   ├── opencv_face_detector.pbtxt"
echo "│   ├── opencv_face_detector_uint8.pb"
echo "│   └── fer2013_mini_XCEPTION.102-0.66.hdf5"
echo "└── sentence_transformers/"
echo "    └── (automatically managed)"
echo

echo "Note: The system works in demo mode without these models."
echo "Real model integration requires downloading and configuring the above files."
echo

echo "Creating model directories..."
mkdir -p ./models/emotion_models
mkdir -p ./models/sentence_transformers

echo "✅ Model directories created. Please download the required model files manually."