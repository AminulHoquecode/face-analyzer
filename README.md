# Face Analyzer

A real-time face analysis application using OpenCV that detects faces and estimates:
- Age
- Gender
- Emotions
- Lighting conditions

The application saves detected faces and their analysis data in a structured format for future reference.

## Features

- Real-time face detection using webcam
- Age estimation in categories (Child, Teen, Young Adult, Adult, Senior)
- Gender prediction based on facial features
- Emotion detection (Happy, Sad, Angry, Surprised, Neutral, Fear, Disgust)
- Lighting condition analysis
- FPS display
- Face tracking for stable detection
- Automatic saving of face images and analysis data

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install opencv-python numpy
   ```

## Usage

Run the script:
```bash
python face_analyzer.py
```

- Press 'q' to quit the application
- Detected faces and their analysis data will be saved in a 'saved_faces' directory

## Structure

The application saves detected faces and their analysis in the following structure:
```
saved_faces/
    YYYY-MM-DD/
        face_YYYYMMDD_HHMMSS_microseconds.jpg
        face_YYYYMMDD_HHMMSS_microseconds.json
```

## License

MIT License