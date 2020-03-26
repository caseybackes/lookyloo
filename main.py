import cv2
import numpy as np
import sys
import os
import argparse


# Haar Cascade models
face_casc_path = '../lookyloo-venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'
eye_casc_path = '../lookyloo-venv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml'

faceCascade = cv2.CascadeClassifier(face_casc_path)
eyeCascade = cv2.CascadeClassifier(eye_casc_path)

def selective_save(img,img_class):
    directory_choices = ['test', 'train']
    selection = directory_choices[np.random.binomial(1, 0.8)]
    sufix = len(os.listdir(f'images/{selection}/{img_class}'))
    cv2.imwrite(f'images/{selection}/{img_class}/eye_image-{sufix}.png', img)



def data_collect(img_class):
    while True:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        # greyscale the frame image for the face object model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        # Draw bounding boxes around the faces
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Draw a rectangle around the eyes
            eyes = eyeCascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                
                # Save the eye image
                eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                selective_save(eye_img, img_class)


        cv2.imshow('Collecting Eye Traker Data',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # When everything is done, release the capture
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect eye-tracker training and test images.')
    parser.add_argument('--direction', '-d', type=str, help='Direction of eye gaze (left or right)')
    args = parser.parse_args()

    data_collect(args.direction)
