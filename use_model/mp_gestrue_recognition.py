import csv
import copy
import itertools

import cv2 as cv
import mediapipe as mp

from keypoint_classifier_ReLU import KeyPointClassifier

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap_width = 960
cap_height = 540

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

keypoint_classifier = KeyPointClassifier()

keypoint_classifier_labels = ["ONE", "TWO", "THREE", "FOUR", "OK", "MENU", "POINTING"]

def calc_landmark_list(landmarks):
    image_width, image_height = cap_width, cap_height

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    origin_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(origin_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        origin_landmark_list[index][0] = origin_landmark_list[index][0] - base_x
        origin_landmark_list[index][1] = origin_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    origin_landmark_list = list(itertools.chain.from_iterable(origin_landmark_list))

    # Normalization
    max_value = max(list(map(abs, origin_landmark_list)))

    def normalize_(n):
        return n / max_value

    origin_landmark_list = list(map(normalize_, origin_landmark_list))

    return origin_landmark_list

while True:
    # Process Key (ESC: end) #################################################
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break
    success, image = cap.read()
    image = cv.flip(image, 1)
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            landmark_list = calc_landmark_list(hand_landmarks)

            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            hand_sign_id, keypoint_classifier_prob = keypoint_classifier(pre_processed_landmark_list)
            keypoint_classifier_prob = '%.2f%%' % (keypoint_classifier_prob * 100)

            cv.putText(image, keypoint_classifier_labels[hand_sign_id], (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3, cv.LINE_AA)
            cv.putText(image, str(keypoint_classifier_prob), (10, 100), cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3, cv.LINE_AA)

    # Flip the image horizontally for a selfie-view display.
    cv.imshow('MediaPipe Hands', image)

cap.release()