#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
import time

print("=" * 60)
print("POINT HISTORY DATA COLLECTOR")
print("Dynamic Finger Gesture Recognition")
print("=" * 60)

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
time.sleep(1)

if not cap.isOpened():
    print("ERROR: Could not open camera!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Test camera
ret, test_frame = cap.read()
for i in range(10):
    ret, test_frame = cap.read()
    if ret:
        break
    time.sleep(0.5)

if not ret:
    print("ERROR: Could not read from camera!")
    cap.release()
    exit(1)

print("Camera initialized!")
print("\nDynamic Gesture Classes:")
print("  0 = Stop (finger stationary)")
print("  1 = Clockwise (draw circle clockwise)")
print("  2 = Counter Clockwise (draw circle counter-clockwise)")
print("  3 = Move (move finger left/right/up/down)")
print("\nControls:")
print("  'h' = Toggle point history logging mode ON/OFF")
print("  '0-3' = Select gesture class")
print("  'n' = Log one sample (press while performing gesture)")
print("  'q' or ESC = Quit")
print("=" * 60)
print("IMPORTANT:")
print("  - Use your INDEX FINGER for all gestures")
print("  - Make slow, deliberate movements")
print("  - Each sample captures your finger movement trail")
print("=" * 60)

# Configuration
history_length = 16  # Number of points to track
point_history = deque(maxlen=history_length)

# State
logging_mode = False
current_class = -1
samples_collected = {0: 0, 1: 0, 2: 0, 3: 0}

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    
    # Flatten
    temp_point_history = list(np.array(temp_point_history).flatten())
    
    return temp_point_history

def logging_csv(number, mode, point_history_data):
    if mode == 0:
        return
    if 0 <= number <= 9:
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([*point_history_data, number])
        return True
    return False

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            # Draw trail with fading effect
            thickness = int((index + 1) / len(point_history) * 6)
            cv2.circle(image, (point[0], point[1]), thickness, (0, 255, 0), -1)
    
    # Draw lines connecting points
    for i in range(len(point_history) - 1):
        if point_history[i][0] != 0 and point_history[i][1] != 0:
            if point_history[i + 1][0] != 0 and point_history[i + 1][1] != 0:
                cv2.line(image, tuple(point_history[i]), tuple(point_history[i + 1]),
                        (0, 255, 0), 2)
    return image

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('h'):
            logging_mode = not logging_mode
            status = "LOGGING MODE ON" if logging_mode else "LOGGING MODE OFF"
            print(f"\n{status}")
            if not logging_mode:
                current_class = -1
                point_history.clear()
        elif logging_mode and ord('0') <= key <= ord('9'):
            current_class = key - ord('0')
            if current_class <= 3:
                class_names = ['Stop', 'Clockwise', 'Counter CW', 'Move']
                print(f"Selected class {current_class}: {class_names[current_class]}")
        elif key == ord('n') and logging_mode and 0 <= current_class <= 3:
            # Log current point history
            if len(point_history) == history_length:
                point_history_data = pre_process_point_history(debug_image, point_history)
                if logging_csv(current_class, 1, point_history_data):
                    samples_collected[current_class] += 1
                    class_names = ['Stop', 'Clockwise', 'Counter CW', 'Move']
                    print(f"  ✅ Logged sample for {class_names[current_class]} (Total: {samples_collected[current_class]})")
            else:
                print(f"  ⚠️ Not enough history points ({len(point_history)}/{history_length})")
        
        # Track index finger tip (landmark 8)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # Add index finger tip to history
                index_finger_tip = landmark_list[8]
                point_history.append(index_finger_tip)
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                )
        else:
            point_history.append([0, 0])
        
        # Draw point history trail
        debug_image = draw_point_history(debug_image, point_history)
        
        # Draw UI
        mode_text = "LOGGING MODE" if logging_mode else "NORMAL MODE"
        mode_color = (0, 255, 0) if logging_mode else (200, 200, 200)
        cv2.putText(debug_image, mode_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        if logging_mode:
            cv2.putText(debug_image, f"History: {len(point_history)}/{history_length}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if 0 <= current_class <= 3:
                class_names = ['Stop', 'Clockwise', 'Counter CW', 'Move']
                cv2.putText(debug_image, f"Class: {current_class} ({class_names[current_class]})", 
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(debug_image, f"Samples: {samples_collected[current_class]}", 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Sample counts
        cv2.putText(debug_image, 
                    f"0:{samples_collected[0]} 1:{samples_collected[1]} 2:{samples_collected[2]} 3:{samples_collected[3]}", 
                    (10, debug_image.shape[0] - 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Press 'h' to toggle logging",
            "Press 0-3 to select class",
            "Press 'n' to log one sample",
            "Press 'q' to quit"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(debug_image, text, 
                        (10, debug_image.shape[0] - 70 + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Point History Data Collector', debug_image)

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    total = sum(samples_collected.values())
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"  Class 0 (Stop): {samples_collected[0]}")
    print(f"  Class 1 (Clockwise): {samples_collected[1]}")
    print(f"  Class 2 (Counter CW): {samples_collected[2]}")
    print(f"  Class 3 (Move): {samples_collected[3]}")
    print(f"\nData saved to: model/point_history_classifier/point_history.csv")
    print("=" * 60)