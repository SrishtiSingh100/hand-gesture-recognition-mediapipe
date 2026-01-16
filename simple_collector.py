#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import cv2
import mediapipe as mp
import time

print("=" * 60)
print("HAND GESTURE DATA COLLECTOR")
print("=" * 60)
print("\nInitializing camera...")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

# IMPORTANT: Give camera time to warm up
print("Warming up camera...")
time.sleep(2)

# Try to read multiple times
ret = False
for i in range(10):
    ret, test_frame = cap.read()
    if ret:
        print(f"Camera ready after {i+1} attempts!")
        break
    time.sleep(0.5)

if not ret:
    print("\nERROR: Could not read from camera after multiple attempts!")
    print("Trying alternative camera index...")
    cap.release()
    
    # Try camera index 1
    cap = cv2.VideoCapture(1)
    time.sleep(2)
    ret, test_frame = cap.read()
    
    if not ret:
        print("ERROR: No working camera found!")
        print("\nTroubleshooting steps:")
        print("1. Check System Preferences > Security & Privacy > Camera")
        print("2. Grant camera access to Terminal/Python")
        print("3. Close other apps using the camera (Zoom, FaceTime, etc.)")
        print("4. Restart your Mac")
        cap.release()
        exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera initialized successfully!")
print("\nGesture Classes:")
print("  0 = Open hand")
print("  1 = Closed fist")
print("  2 = Pointing finger")
print("  3 = OK sign")
print("\nControls:")
print("  'k' = Toggle logging mode ON/OFF")
print("  '0-3' = Select gesture class")
print("  'q' or ESC = Quit")
print("=" * 60)
print("Camera window should appear now...")

logging_mode = False
current_class = -1
samples_collected = {0: 0, 1: 0, 2: 0, 3: 0}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('k'):
            logging_mode = not logging_mode
            status = "LOGGING MODE ON" if logging_mode else "LOGGING MODE OFF"
            print(f"\n{status}")
            if not logging_mode:
                current_class = -1
        elif logging_mode and ord('0') <= key <= ord('3'):
            current_class = key - ord('0')
            class_names = ['Open', 'Close', 'Pointer', 'OK']
            print(f"Selected class {current_class}: {class_names[current_class]}")
        
        if results.multi_hand_landmarks and logging_mode and 0 <= current_class <= 3:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)
                
                base_x, base_y = landmarks[0], landmarks[1]
                normalized = []
                for i in range(0, len(landmarks), 2):
                    normalized.append(landmarks[i] - base_x)
                    normalized.append(landmarks[i+1] - base_y)
                
                max_val = max(abs(x) for x in normalized)
                if max_val > 0:
                    normalized = [x / max_val for x in normalized]
                
                csv_path = 'model/keypoint_classifier/keypoint.csv'
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(normalized + [current_class])
                
                samples_collected[current_class] += 1
                
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        
        mode_text = "LOGGING MODE" if logging_mode else "NORMAL MODE"
        mode_color = (0, 255, 0) if logging_mode else (200, 200, 200)
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        if logging_mode and 0 <= current_class <= 3:
            class_names = ['Open', 'Close', 'Pointer', 'OK']
            cv2.putText(frame, f"Class: {current_class} ({class_names[current_class]})", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Samples: {samples_collected[current_class]}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, f"0:{samples_collected[0]} 1:{samples_collected[1]} 2:{samples_collected[2]} 3:{samples_collected[3]}", 
                    (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'k' to toggle logging", 
                    (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 0-3 to select class", 
                    (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Hand Gesture Data Collector', frame)

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"\nERROR: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    total = sum(samples_collected.values())
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"  Class 0 (Open): {samples_collected[0]}")
    print(f"  Class 1 (Close): {samples_collected[1]}")
    print(f"  Class 2 (Pointer): {samples_collected[2]}")
    print(f"  Class 3 (OK): {samples_collected[3]}")
    print(f"\nData saved to: model/keypoint_classifier/keypoint.csv")