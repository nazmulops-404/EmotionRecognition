# Emotion Recognition from Speech (RAVDESS + MFCC + CNN/LSTM)

CodeAlpha Machine Learning Internship, Task 2.

## Objective
Recognize human emotion (angry, happy, sad and others) from a short speech recording.

## Dataset
RAVDESS, speech part (downloaded from Zenodo).
* 1,440 audio files from 24 actors (12 male, 12 female)
* Each actor says two sentences in 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
* The emotion and actor are read from each file name

## Method
1. Loaded each audio file (first 0.5 seconds skipped, then 3 seconds kept).
2. Extracted 40 MFCC coefficients per time frame. Every file became a 130 x 40 matrix.
3. Split the data by actor: actors 1 to 20 for training (1,200 files) and actors 21 to 24 for testing (240 files). The same actor never appears in both sets, so the model cannot memorize a voice.
4. Standardized the features using training data only.
5. Trained two models with Adam and early stopping (patience 10, best weights restored, 15% of training data for validation):
   * CNN: two Conv1D layers (64 and 128 filters) with batch normalization, pooling and dropout
   * LSTM: one LSTM layer with 128 units and dropout
6. Evaluated on the 240 test files with Accuracy, Precision, Recall, F1 Score and a Confusion Matrix.

## Results

| Model | Test Accuracy | Epochs run |
|---|---|---|
| CNN | 42.92% | 17 |
| LSTM | 35.42% | 25 |

With 8 classes, random guessing gives about 12.5%. Both models are well above that, but neither is reliable.

CNN, per emotion:

| Emotion | Precision | Recall | F1 Score | Test files |
|---|---|---|---|---|
| angry | 0.490 | 0.750 | 0.593 | 32 |
| calm | 0.731 | 0.594 | 0.655 | 32 |
| disgust | 0.750 | 0.188 | 0.300 | 32 |
| fearful | 0.667 | 0.438 | 0.528 | 32 |
| happy | 0.235 | 0.500 | 0.320 | 32 |
| neutral | 0.500 | 0.125 | 0.200 | 16 |
| sad | 0.288 | 0.531 | 0.374 | 32 |
| surprised | 1.000 | 0.156 | 0.270 | 32 |

Overall accuracy 0.429 (103 of 240 correct), macro average F1 0.405.

## Observations
* The CNN beat the LSTM, so it is the final model.
* Angry (recall 0.75), calm (0.59) and sad (0.53) were recognized best.
* Neutral (recall 0.125), surprised (0.156) and disgust (0.188) were recognized worst.
* The confusion matrix shows many wrong guesses going to "happy" and "sad". For example, 15 of 32 fearful files and 12 of 32 surprised files were predicted as happy, and 12 of 32 calm files were predicted as sad.
* The training curves show strong overfitting. Training accuracy reached about 0.9, while validation accuracy stayed between roughly 0.3 and 0.4.

## Limitations
* Only 1,200 training files from 20 actors is small for this task, and the models overfit.
* The test set has only 4 actors, so the score would change with different actors.
* RAVDESS is acted speech with two fixed sentences in English. The model is not expected to work on natural speech or other languages.
* This is a learning project and not a reliable emotion detector.

## Ideas to improve (not tested)
* Data augmentation (added noise, pitch shift, time stretch)
* More features besides MFCC, such as mel spectrograms or chroma
* More data from TESS or EMO DB
* Stronger regularization

## How to run
1. Open the notebook file in Google Colab. An internet connection is needed. The dataset (about 215 MB) downloads automatically.
2. Click Runtime, then Run all.
3. The last cell asks for an audio file of your own voice. It is optional.

## Tools
Python, Librosa, TensorFlow/Keras, NumPy, Pandas, Scikit-learn, Matplotlib
