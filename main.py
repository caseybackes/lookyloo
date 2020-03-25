import cv2
import numpy as np
import sys

face_casc_path = '../lookyloo-venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'
eye_casc_path = '../lookyloo-venv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml'


faceCascade = cv2.CascadeClassifier(face_casc_path)
eyeCascade = cv2.CascadeClassifier(eye_casc_path)

if __name__ == "__main__":

    while True:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        # greyscale the frame image for the face object model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around the faces
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Draw a rectangle around the eyes
            eyes = eyeCascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                eyes_median_area = np.median(eyes[:, 2])
                for (ex,ey,ew,eh) in eyes:
                    # only the two largest eye boxes
                    eye_area = ew
                    if abs(eye_area - eyes_median_area)/eyes_median_area <= 0.2: 
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('img',frame)
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # When everything is done, release the capture
            cap.release()
            cv2.destroyAllWindows()
            break
