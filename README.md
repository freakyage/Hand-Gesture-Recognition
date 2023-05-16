# Hand-Gesture-Recognition
Using Scikit-Learn's RandomizedSearchCV to train the model and using the trained model for gesture recognition.

Detecting hand landmarks using [MediaPipe Hands](https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md), collecting them as training data, and recording the landmarks as a keypoint.csv file.
Assigning the gestures from "0" to "9" (from ***"Top Row Numeric Keys"***) to represent different hand gestures and recording them accordingly.
```bash
python collect_keypoints/mp_collect_keypoint.py
```

Please open and run the "/training_model/keypoint_classification_ReLU_scikeras.ipynb" file using [Jupyter Notebook](https://jupyter.org/).

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
