import cv2 as cv
import imutils
import numpy as np
import keras

# define all constants
model_path = "./model/CNN-Emotion-Model"
face_path = cv.haarcascades + 'haarcascade_frontalface_alt.xml'
emotion_label = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# create video capture object
cap = cv.VideoCapture(0)
# load face detection model
face_model = cv.CascadeClassifier()
if not face_model.load(face_path):
    print('load face model failed')
    exit(0)

# load trained emotion model
emotion_model = keras.models.load_model(model_path)

def drawFace(frame, frame_grey):
    faces = face_model.detectMultiScale(frame, minSize=(50,50))
    emotion = np.array([[0,0,0,0,0,0,0]])
    for a,b,c,d in faces:
        # print('%s %s %s %s' % (a,b,c,d))
        face = frame_grey[b:b+d,a:a+c]
        face = cv.resize(face, (48, 48))

        face = np.array(face)
        face = face.reshape(1, 48, 48, 1)

        emotion = emotion_model.predict(face/255)
        text = emotion_label[np.argmax(emotion)]

        cv.rectangle(frame, (a,b), (a+c, b+d), (0, 0, 255), 1)
        cv.putText(frame, text, (a,b-10),cv.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)
    return emotion

def main():
    # run real time camera
    while True:
        # if failed to open camera
        if not cap.isOpened():
            print('open camera failed')
            break
        # read current frame image
        frame = cap.read()[1]
        frame = imutils.resize(frame, width=300)
        # construct prob canvas
        prob = np.ones((300, 500, 3), dtype='uint8')

        # transfer to grey image
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        emotion = drawFace(frame, frame_grey)
        # print(emotion)
        sum = emotion.sum()
        if not sum == 0:
            emotion = emotion / sum

        # print(emotion)
        for i in range(0, 7):
            cv.putText(prob, emotion_label[i], (10, (i + 1) * 35 + 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
                       1)
            lr = 90 + int(emotion[0][i] * 300)
            cv.rectangle(prob, (90, i * 35 + 25), (90 + int(emotion[0][i] * 300), (i + 1) * 35 + 30), (0, 255, 0), -1)
            cv.putText(prob, str(int(emotion[0][i] * 100)) + '%', (lr, (i + 1) * 35 + 10), cv.FONT_HERSHEY_COMPLEX, 0.5,
                       (255, 255, 255), 1)

        cv.imshow('probability', prob, )
        cv.imshow('video', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()
