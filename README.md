# Hand-Gesture-Recognition
Using Scikit-Learn's RandomizedSearchCV to train the model and using the trained model for gesture recognition.

## Step 1.
Detecting hand landmarks using [MediaPipe Hands](https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md), collecting them as training data, and recording the landmarks as a keypoint.csv file.
Assigning the gestures from "0" to "9" (from ***"Top Row Numeric Keys"***) to represent different hand gestures and recording them accordingly.
```bash
python collect_keypoints/mp_collect_keypoint.py
```
## Step 2.
Please open "keypoint_classification_ReLU_scikeras.ipynb" using [Jupyter Notebook](https://jupyter.org/) and run it to train the model.

## Step 3.
Please replace the variable "keypoint_classifier_labels" with the names of the hand gestures you have trained for hand gesture recognition **in line 29 of the "mp_gesture_recognition.py" file**.

Run "mp_gesture_recognition.py" to perform hand gesture recognition.
```bash
python mp_gestrue_recognition.py
```

## Requirements
- TensorFlow >= 2.7
- mediapipe
- opencv-python
- scikit-learn
- matplotlib
- pandas
- pytz
- scikeras
- seaborn
- xlwt
```bash
pip install -r requirements.txt
```

## Reference
- [kinivi/hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe)
