import cv2
import time
import numpy as np
import os
from datetime import datetime
import json

def setup_save_directory():
    """Create directories for saving faces and their details"""
    # Create base directory for saved faces
    save_dir = "saved_faces"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create directory with today's date
    today = datetime.now().strftime("%Y-%m-%d")
    today_dir = os.path.join(save_dir, today)
    if not os.path.exists(today_dir):
        os.makedirs(today_dir)
    
    return today_dir

def estimate_age(feature_history):
    """Estimate approximate age range based on face features using a weighted scoring system"""
    if not feature_history:
        return "Unknown"
    
    # Get the most recent features and calculate averages
    recent_features = feature_history[-1]
    
    # Use more frames (10) for better stability in age estimation
    if len(feature_history) >= 10:
        avg_features = {}
        for key in ['contrast', 'var_r', 'var_g', 'var_b', 'eye_distance', 'face_ratio']:
            values = [f.get(key, 0) for f in feature_history[-10:]]
            # Remove outliers before averaging
            values.sort()
            trimmed_values = values[2:-2]  # Remove 2 highest and 2 lowest values
            avg_features[key] = sum(trimmed_values) / len(trimmed_values)
    else:
        avg_features = recent_features
    
    # Extract and normalize features
    contrast = avg_features.get('contrast', 0)
    eye_distance = avg_features.get('eye_distance', 0)
    face_width = recent_features.get('face_width', 1)
    face_ratio = avg_features.get('face_ratio', 0)
    texture_variation = (avg_features.get('var_r', 0) + 
                        avg_features.get('var_g', 0) + 
                        avg_features.get('var_b', 0)) / 3
    
    # Calculate relative eye distance to face width
    rel_eye_distance = eye_distance / face_width if face_width > 0 else 0
    
    # Initialize age scores
    age_scores = {
        "Child": 0,
        "Teen": 0,
        "Young Adult": 0,
        "Adult": 0,
        "Senior": 0
    }
    
    # Score based on contrast (skin smoothness)
    if contrast > 55: age_scores["Child"] += 3
    elif contrast > 50: age_scores["Teen"] += 2.5
    elif contrast > 45: age_scores["Young Adult"] += 2
    elif contrast > 40: age_scores["Adult"] += 2
    else: age_scores["Senior"] += 2.5

    # Score based on relative eye distance
    if rel_eye_distance > 0.34: age_scores["Child"] += 2.5
    elif rel_eye_distance > 0.32: age_scores["Teen"] += 2
    elif rel_eye_distance > 0.30: age_scores["Young Adult"] += 1.5
    elif rel_eye_distance > 0.28: age_scores["Adult"] += 1.5
    else: age_scores["Senior"] += 1

    # Score based on face ratio
    if face_ratio < 0.82: age_scores["Child"] += 2
    elif 0.82 <= face_ratio < 0.85: age_scores["Teen"] += 2
    elif 0.85 <= face_ratio < 0.88: age_scores["Young Adult"] += 2
    elif 0.88 <= face_ratio < 0.91: age_scores["Adult"] += 2
    else: age_scores["Senior"] += 1.5

    # Score based on texture variation
    if texture_variation < 1800: age_scores["Child"] += 3
    elif texture_variation < 2300: age_scores["Teen"] += 2.5
    elif texture_variation < 2800: age_scores["Young Adult"] += 2
    elif texture_variation < 3500: age_scores["Adult"] += 2
    else: age_scores["Senior"] += 3

    # Get the age group with highest score
    max_score = max(age_scores.values())
    age_group = max(age_scores.items(), key=lambda x: x[1])[0]
    
    
    # Only return confident predictions
    if max_score < 4:
        return "Unknown"
        
    # Map age group to range
    age_ranges = {
        "Child": "Child (0-12)",
        "Teen": "Teen (13-19)",
        "Young Adult": "Young Adult (20-35)",
        "Adult": "Adult (36-50)",
        "Senior": "Senior (50+)"
    }
    
    return age_ranges[age_group]

def save_face(face_img, details, save_dir):
    """Save face image and its details"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Save the face image
    image_filename = f"face_{timestamp}.jpg"
    image_path = os.path.join(save_dir, image_filename)
    cv2.imwrite(image_path, face_img)
    
    # Save the details in a JSON file
    details_filename = f"face_{timestamp}.json"
    details_path = os.path.join(save_dir, details_filename)
    
    # Add timestamp to details
    details['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    with open(details_path, 'w') as f:
        json.dump(details, f, indent=4)

def main():
    print("Starting Enhanced Face Detection Application...")
    print("Press 'q' to quit")
    
    # Setup tracking variables
    TRACKING_THRESHOLD = 50  # pixels
    previous_faces = []
    MAX_TRACKING_HISTORY = 5
    
    # Setup save directory
    save_dir = setup_save_directory()
    print(f"Saving faces to: {save_dir}")
    
    # Setup face detection
    face_cascade_path = setup_face_detector()
    if not face_cascade_path:
        print("Error: Could not set up face detection. Exiting.")
        return
    
    # Load the cascade classifiers
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Setup eye detector (useful for gender estimation heuristics)
    eye_cascade_path = setup_eye_detector()
    eye_cascade = None
    if eye_cascade_path:
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    # Try to start webcam
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        return
    
    # Initialize variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Initialize prediction variables
    emotion_prediction = "Unknown"
    gender_prediction = "Unknown"
    
    # Feature tracking variables
    face_features_history = []
    max_history = 10  # Keep track of recent features for smoother predictions
    
    # Add a counter to cycle through emotions for testing
    test_mode = False  # Set to True to cycle through emotions for testing
    test_counter = 0
    test_emotions = ["Happy", "Sad", "Angry", "Surprised", "Neutral", "Fear", "Disgust"]
    
    while True:
        ret, frame = cap.read()  # Capture frame
        if not ret:
            print("Error: Failed to capture image from camera")
            break
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert faces to list for tracking
        current_faces = []
        for (x, y, w, h) in faces:
            current_faces.append((x, y, w, h))
        
        # If no faces detected, try to use previous face locations
        if len(current_faces) == 0 and len(previous_faces) > 0:
            # Use the most recent previous face location
            last_face = previous_faces[-1]
            # Expand search area slightly
            search_x = max(0, last_face[0] - 20)
            search_y = max(0, last_face[1] - 20)
            search_w = min(frame.shape[1] - search_x, last_face[2] + 40)
            search_h = min(frame.shape[0] - search_y, last_face[3] + 40)
            
            # Try detection in the smaller region
            search_roi = gray[search_y:search_y+search_h, search_x:search_x+search_w]
            if search_roi.size > 0:
                local_faces = face_cascade.detectMultiScale(
                    search_roi,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(30, 30)
                )
                
                # Adjust coordinates back to full frame
                for (fx, fy, fw, fh) in local_faces:
                    current_faces.append((fx + search_x, fy + search_y, fw, fh))
        
        # Filter faces based on previous locations
        filtered_faces = []
        if len(previous_faces) > 0:
            for current_face in current_faces:
                x, y, w, h = current_face
                current_center = (x + w//2, y + h//2)
                
                # Check if this face is near any previous face
                for prev_face in previous_faces[-1:]:
                    px, py, pw, ph = prev_face
                    prev_center = (px + pw//2, py + ph//2)
                    
                    distance = np.sqrt(
                        (current_center[0] - prev_center[0])**2 +
                        (current_center[1] - prev_center[1])**2
                    )
                    
                    if distance < TRACKING_THRESHOLD:
                        # Smooth the transition
                        smoothed_x = int(0.7 * px + 0.3 * x)
                        smoothed_y = int(0.7 * py + 0.3 * y)
                        smoothed_w = int(0.7 * pw + 0.3 * w)
                        smoothed_h = int(0.7 * ph + 0.3 * h)
                        filtered_faces.append((smoothed_x, smoothed_y, smoothed_w, smoothed_h))
                        break
                else:
                    filtered_faces.append(current_face)
        else:
            filtered_faces = current_faces
        
        # Update tracking history
        if len(filtered_faces) > 0:
            previous_faces.append(filtered_faces[0])  # Track the first face
            if len(previous_faces) > MAX_TRACKING_HISTORY:
                previous_faces.pop(0)
        
        # Process each detected face
        for (x, y, w, h) in filtered_faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            # Extract face features for analysis
            features = extract_face_features(face_img, face_gray, eye_cascade)
            
            # Add to history
            face_features_history.append(features)
            if len(face_features_history) > max_history:
                face_features_history.pop(0)
            
            # Estimate emotion and gender based on features
            detected_emotion = estimate_emotion(face_features_history)
            
            # If in test mode, cycle through emotions (for testing only)
            if test_mode:
                if frame_count % 30 == 0:  # Change every 30 frames
                    test_counter = (test_counter + 1) % len(test_emotions)
                emotion_prediction = test_emotions[test_counter]
            else:
                emotion_prediction = detected_emotion
                
            gender_prediction = estimate_gender(face_features_history)
            age_prediction = estimate_age(face_features_history)
            
            # Calculate lighting
            brightness = np.mean(face_img)
            lighting = "Normal"
            if brightness > 150:
                lighting = "Bright"
            elif brightness < 100:
                lighting = "Dark"
            
            # Save face and details
            details = {
                'emotion': emotion_prediction,
                'gender': gender_prediction,
                'age': age_prediction,
                'lighting': lighting,
                'features': {
                    'face_width': features.get('face_width'),
                    'face_height': features.get('face_height'),
                    'face_ratio': features.get('face_ratio'),
                    'eye_count': features.get('eye_count'),
                    'eye_distance': features.get('eye_distance')
                }
            }
            save_face(face_img, details, save_dir)
            
            # Display face information
            cv2.putText(frame, f"Face Detected", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display analysis results
            y_offset = h + 25
            cv2.putText(frame, f"Emotion: {emotion_prediction}", (x, y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Gender: {gender_prediction}", (x, y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add age display
            y_offset += 25
            cv2.putText(frame, f"Age: {age_prediction}", (x, y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display lighting
            y_offset += 25
            cv2.putText(frame, f"Lighting: {lighting}", (x, y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # If no faces were detected
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Position yourself in front of camera", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Face Analysis', frame)
        
        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    print("Closing application...")
    cap.release()
    cv2.destroyAllWindows()

def extract_face_features(face_img, face_gray, eye_cascade):
    """Extract features from a face image for analysis"""
    height, width = face_gray.shape
    features = {}
    
    # Basic size and shape features
    features['face_width'] = width
    features['face_height'] = height
    features['face_ratio'] = width / height if height > 0 else 0
    
    # Calculate average color (different channels can indicate skin tone)
    if len(face_img.shape) == 3:  # Color image
        features['avg_r'] = np.mean(face_img[:,:,2])
        features['avg_g'] = np.mean(face_img[:,:,1])
        features['avg_b'] = np.mean(face_img[:,:,0])
        
        # Color variance can indicate facial features/expressions
        features['var_r'] = np.var(face_img[:,:,2])
        features['var_g'] = np.var(face_img[:,:,1])
        features['var_b'] = np.var(face_img[:,:,0])
    
    # Texture features
    features['contrast'] = np.std(face_gray)
    
    # Detect eyes if available
    eyes = []
    if eye_cascade is not None:
        eyes = eye_cascade.detectMultiScale(face_gray)
    
    features['eye_count'] = len(eyes)
    
    # Eye features if available
    if len(eyes) >= 2:
        # Sort eyes by x-coordinate
        eyes = sorted(eyes, key=lambda eye: eye[0])
        
        # Calculate eye distance and positions
        eye1_x, eye1_y, eye1_w, eye1_h = eyes[0]
        eye2_x, eye2_y, eye2_w, eye2_h = eyes[1]
        
        eye1_center = (eye1_x + eye1_w//2, eye1_y + eye1_h//2)
        eye2_center = (eye2_x + eye2_w//2, eye2_y + eye2_h//2)
        
        features['eye_distance'] = np.sqrt((eye2_center[0] - eye1_center[0])**2 + 
                                          (eye2_center[1] - eye1_center[1])**2)
        features['eye_y_diff'] = abs(eye1_center[1] - eye2_center[1])
    else:
        features['eye_distance'] = 0
        features['eye_y_diff'] = 0
    
    return features

def estimate_emotion(feature_history):
    """Estimate emotion based on face features"""
    if not feature_history:
        return "Unknown"
    
    # Get the most recent features
    recent_features = feature_history[-1]
    
    # Get average features from history for more stability
    if len(feature_history) >= 3:
        # Calculate average of last 3 frames to reduce flickering
        avg_features = {}
        for key in ['contrast', 'var_r', 'var_g', 'var_b', 'eye_y_diff']:
            values = [f.get(key, 0) for f in feature_history[-3:]]
            avg_features[key] = sum(values) / len(values)
    else:
        avg_features = recent_features
    
    # Extract key features for emotion detection
    contrast = avg_features.get('contrast', 0)
    var_r = avg_features.get('var_r', 0)
    var_g = avg_features.get('var_g', 0)
    var_b = avg_features.get('var_b', 0)
    eye_y_diff = avg_features.get('eye_y_diff', 0)
    
    # Debug values (uncomment for debugging)
    # print(f"Contrast: {contrast:.2f}, Var_R: {var_r:.2f}, Eye Y-diff: {eye_y_diff:.2f}")
    
    # More nuanced emotion detection logic
    
    # Very high contrast with aligned eyes often indicates surprise
    if contrast > 60 and eye_y_diff < 4:
        return "Surprised"
    
    # High red variance with moderate contrast often indicates happiness
    # Increased threshold to make it less likely to trigger
    if var_r > 3000 and contrast > 45:
        return "Happy"
    
    # High contrast with eye misalignment often indicates anger
    if contrast > 55 and eye_y_diff > 5:
        return "Angry"
    
    # Low contrast combined with blue/green dominance often indicates sadness
    if contrast < 40 and var_b > var_r:
        return "Sad"
    
    # High green variance can indicate disgust
    if var_g > var_r * 1.2 and var_g > var_b * 1.2:
        return "Disgust"
    
    # High blue variance with low contrast can indicate fear
    if var_b > var_r * 1.1 and contrast < 45:
        return "Fear"
    
    # Moderate values tend to indicate neutral
    if 40 <= contrast <= 55 and var_r < 2500 and var_g < 2500 and var_b < 2500:
        return "Neutral"
    
    # Default to Neutral if no specific emotion is detected
    return "Neutral"

def estimate_gender(feature_history):
    """Estimate gender based on face features"""
    if not feature_history:
        return "Unknown"
    
    # Get average features from history for more stability
    avg_face_ratio = sum(f.get('face_ratio', 0) for f in feature_history) / len(feature_history)
    avg_eye_distance = sum(f.get('eye_distance', 0) for f in feature_history) / len(feature_history)
    
    # Very simple heuristic based on face shape
    # Note: This is a rough approximation and not accurate for all faces
    # Males often have wider faces relative to height
    if avg_face_ratio > 0.85:
        return "Male"
    elif avg_face_ratio < 0.78:
        return "Female"
    
    # Eye distance relative to face size can be another indicator
    # Use the most recent face width for scaling
    face_width = feature_history[-1].get('face_width', 1)
    relative_eye_distance = avg_eye_distance / face_width if face_width > 0 else 0
    
    if relative_eye_distance > 0.35:
        return "Female"
    else:
        return "Male"
    
    # This is a simplistic approach - in reality, gender detection requires more sophisticated analysis

def setup_face_detector():
    """Download and set up the face detection cascade classifier"""
    # Path to the Haar cascade file
    cascade_file = 'haarcascade_frontalface_default.xml'
    
    # Check if the file already exists
    if os.path.isfile(cascade_file):
        print(f"Using existing cascade file: {cascade_file}")
        return cascade_file
    
    # Check if it's in the OpenCV installation
    opencv_cascade_path = os.path.join(cv2.__path__[0], 'data', cascade_file)
    if os.path.isfile(opencv_cascade_path):
        print(f"Using OpenCV's cascade file: {opencv_cascade_path}")
        return opencv_cascade_path
    
    # Try to download the cascade file
    print("Downloading face detection model...")
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_file)
        print(f"Downloaded cascade file to: {cascade_file}")
        return cascade_file
    except Exception as e:
        print(f"Error downloading cascade file: {str(e)}")
        return None

def setup_eye_detector():
    """Set up eye detection cascade classifier"""
    # Path to the Haar cascade file for eyes
    cascade_file = 'haarcascade_eye.xml'
    
    # Check if the file already exists
    if os.path.isfile(cascade_file):
        print(f"Using existing eye cascade file: {cascade_file}")
        return cascade_file
    
    # Check if it's in the OpenCV installation
    opencv_cascade_path = os.path.join(cv2.__path__[0], 'data', cascade_file)
    if os.path.isfile(opencv_cascade_path):
        print(f"Using OpenCV's eye cascade file: {opencv_cascade_path}")
        return opencv_cascade_path
    
    # Try to download the cascade file
    print("Downloading eye detection model...")
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
        urllib.request.urlretrieve(url, cascade_file)
        print(f"Downloaded eye cascade file to: {cascade_file}")
        return cascade_file
    except Exception as e:
        print(f"Error downloading eye cascade file: {str(e)}")
        return None

if __name__ == "__main__":
    main()