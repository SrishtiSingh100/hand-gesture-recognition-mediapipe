#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-Time Hand Gesture Recognition Application
Uses both Keypoint Classifier (static gestures) and Point History Classifier (dynamic gestures)
"""
import csv
import copy
import itertools
from collections import deque, Counter
import cv2
import numpy as np
import mediapipe as mp
import time

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier
from utils.cvfpscalc import CvFpsCalc


def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    
    # Wait for camera to initialize
    time.sleep(1)
    for _ in range(10):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.5)
    
    if not ret:
        print("ERROR: Could not access camera!")
        return

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Load classifiers
    try:
        keypoint_classifier = KeyPointClassifier()
        print("✅ Keypoint classifier loaded")
    except Exception as e:
        print(f"⚠️  Could not load keypoint classifier: {e}")
        keypoint_classifier = None
    
    try:
        point_history_classifier = PointHistoryClassifier()
        print("✅ Point history classifier loaded")
    except Exception as e:
        print(f"⚠️  Could not load point history classifier: {e}")
        point_history_classifier = None

    # Load labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    
    with open('model/point_history_classifier/point_history_classifier_label.csv',
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS calculator
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Point history
    history_length = 16
    point_history = deque(maxlen=history_length)
    
    # Finger gesture history for smoothing
    finger_gesture_history = deque(maxlen=history_length)

    print("\n" + "=" * 60)
    print("HAND GESTURE RECOGNITION - RUNNING")
    print("=" * 60)
    print("Controls:")
    print("  ESC or 'q' - Quit")
    print("=" * 60)

    while True:
        fps = cvFpsCalc.get()

        # Key input
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            continue
            
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                # Calculate landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Normalize coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Normalize point history
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Hand sign classification
                hand_sign_id = 0
                if keypoint_classifier:
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2) and point_history_classifier:
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Add to gesture history for smoothing
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # Draw landmarks
                debug_image = draw_landmarks(debug_image, landmark_list)
                
                # Draw info
                debug_image = draw_info_text(
                    debug_image,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]] if most_common_fg_id else "",
                )
        else:
            point_history.append([0, 0])

        # Draw point history
        debug_image = draw_point_history(debug_image, point_history)
        
        # Draw FPS
        debug_image = draw_info(debug_image, fps)

        # Display
        cv2.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Flatten
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalize
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                         base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                         base_y) / image_height

    # Flatten
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def draw_landmarks(image, landmark_point):
    # Connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]

    # Draw connections
    for connection in connections:
        cv2.line(image, tuple(landmark_point[connection[0]]),
                tuple(landmark_point[connection[1]]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[connection[0]]),
                tuple(landmark_point[connection[1]]), (255, 255, 255), 2)

    # Draw keypoints
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info_text(image, handedness, hand_sign_text, finger_gesture_text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 80), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ': ' + hand_sign_text
    
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)

    if finger_gesture_text != "":
        cv2.putText(image, "Finger Gesture: " + finger_gesture_text, (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def draw_info(image, fps):
    cv2.putText(image, "FPS:" + str(fps), (10, image.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image


if __name__ == '__main__':
    main()