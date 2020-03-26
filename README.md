# LookyLoo
__A project in real-time image recognition and classification. __

## Motivation
About a year ago, I found myself back in the job market looking for my next opportunity. A mentor, and dear friend, Ravin, shared some information with me regarding my resume. The most interesting thing Ravin shared with me was the work that had been done with eye-tracking as applied to resumes. It was basically images of resumes with heat map overlays that represented the amount of time an HR staffer would spend looking at various parts of the resume. Sure, I learned to prioritize certain parts of my resume with this information but looking back (now that I'm a "big deal") I really want to see the visceral innards of the project that produced that information - the eye tracking program. One way to do it would be to crawl the interwebs and find some version of that sort of work, but that's like looking at the answers before taking the exam. I'd really rather build that solution/project myself, researching what I need to on object detection and data acquisition along the way – cementing my own understanding (and probable struggles) on machine learning as a part of a larger application. 

## Objectives 
I think it would damn cool for my machine to know exactly where I’m looking and put a marker on that spot on my screen. Maybe even do some simple trick when I blink (which feels pretty ambitious right now if I’m being honest). Can my program interact with my MacOSX and steer the mouse on my screen based on tracking my eye movements? Can I select applications from the taskbar by “left blinking”? In pursuit of this ambitious __vision__ ( see what I did there ;) I’m sure I’ll learn a great deal about real time image processing and object detection, and likely get more exposure to training models on a cloud-hosted machine. Maybe make a simple game of it and host it as a webapp via Flask or Django (difficulty of “Cake” and “Labyrinth”, respectively). 


## Project Plan v0.0
I feel like a first approximation of the project plan can come from working backwards from object to specific tasks and tools. 
Lets say we have a basic minimum viable project that can output “left” and “right” depending on which side of the computer screen you are looking at (assuming a laptop and built-in webcam). Just a basic print out of “left side of screen” or “right side of screen”. To accomplish this, the model needs to understand where your eyes are looking at the screen. Which implies we have the eyes isolated in the real-time image frames, further implying object detection. But eyes are a very small part of even a webcam image, and many other things could accidently be classified a as eyeballs that are not. So maybe having a facial recognition model that can isolate the face region of the image, and then an object detection model for eyes to isolate just the image(s) of each eye, we can then train a supervised model to classify eye orientation at the granularity of “looking left” or “looking right”. Its now obvious we need image of eyeballs looking at a screen. While there will NEVER be a shortage of such data thanks to smart phones and YouTube, lets take the approach of using the built-in webcam, for two reasons. 1) This is the data that will be used to make predictions later in deployment. 2) There is significantly less to do in the way of data processing and formatting when using one source of data collection. 
So how do we use the webcam to get images that will be used for training? We do have a couple of good options, though the OpenCV2 library and Scikit-Image are the choices I see as immediately obvious. 
Great! Now were all the way down the reverse-engineering path to the point where we’re looking at tool selection. Lets structure this plan back up from beginning to end…

1)	OpenCV2/Scikit-Image to collect images from webcam
2)	Object detection for faces and eyes. 
3)	Train a model to take two images of eyes and head location to predict the side of the screen we are looking at. 
4)	Output to terminal which side of the screen the model predicts we are looking at. 

I feel like one is the only easy step here; the others are going to be a challenge. Fun, but challenging. 


## Step Zero – The Coding Environment
When starting a new project, I like to keep version control top of mind. There will be many iterations to this project as it develops, making version control a critical component to this project. Let’s start a new (empty) repo on GitHub, clone it to my local machine, and make the first branch called `starting-out`. 

    $ git clone <my_project_repo>.git
    $ git branch starting-out


After making an empty file called `main.py`, I’ll open up a terminal and spin up a virtual python environment specifically for this project. I like working with Python3.6+ so lets get it running and start installing the packages we KNOW we need immediately. 

    $ pip install matplotlib numpy keras scikit-image opencv-python-headless

Where `opencv-python-headless` was the necessary version to use after trying `opencv-python` and getting an error regarding dependancies for platforms besides Mac. 

## Data Acquisition v0.0
Some startr code is given in the OpenCV Python documentation tutorial, ["Face Detection using Haar Cascades"](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection). One significant difference here is that we are using a webcam, and the tutorial is using a static image. So let's modify the starter code to leverage the webcam, as well as greyscale the captured video frames for the HaarCascade model. 

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # greyscale the frame image for the face object model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

At this point of the algorithm, we have faces detected in the webcam images. Somthing to know ahead of time is the object representation of the detected face(s). The faces object is represented as an _n x 4_ array, with _n_ being the number of faces detected. For example :

    >>> faces
    array([[576, 345, 241, 241]], dtype=int32)
    
representing the coordinates of the top left corner of the bounding box (576, 345) and the width and height of the box (241,241). We can use this information ahead of time to recolor the outermost pixels of the box to blue with the `cv2.rectangle` function acting on the `frame` object of pixel arrays: 

    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

Here we have modified the captured `frame` to include a blue bounding box and saved this as the variable `img` - which will be useful for reasons we'll see in a moment. We also define a region of interest which contains only the face in the frame. The `roi_gray` is then passed to the eye detection model, and `roi_color` will be used for drawing bounding boxes around the eyes. 

    eyes = eyeCascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

We can then show the whole image and the bounding boxes with the `cv2.imshow` function, passing the name of the window and the frame that has bounding boxes drawn directly on to it.  

    cv2.imshow('uncreative name of window', frame)

Though, we do need to stop the webcam stream at some point, and to do this we designate a "waitKey". Basically, while streaming the webcam video we can press a particular key to activate a release of the camera (kill the stream). Let's set that key to `q` which will release the camera and close the webcam streaming window on our screen. 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
 
With these essential code components, we can build a function to run from the `main.py` file. Let's put the above code into a `while` loop so that we can capture more than a single frame, effectively streaming the webcam data and detecting faces and eyes on every frame. 

    import cv2
    import numpy as np
    import sys

    # Haar Cascade models
    face_casc_path = '../lookyloo-venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'
    eye_casc_path = '../lookyloo-venv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml'

    faceCascade = cv2.CascadeClassifier(face_casc_path)
    eyeCascade = cv2.CascadeClassifier(eye_casc_path)

    if __name__ == "__main__":
        # def collect_eye_images(look_direction = 'left', num = 200)
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
                if len(eyes) >=2:
                    eyes_median_area = np.median(eyes[:, 2])
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            cv2.imshow('LookyLoo',frame)

            # Kill the stream on entering 'q' while on the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # When everything is done, release the capture
                cap.release()
                cv2.destroyAllWindows()
                break

This gives a live stream of the webcam with bounding box overlay of the detected face(s) and eye(s). It is likely that you will blink while looking at the screen, and the model does not detect eyes very well during a blink. 

We also want to save the images of the eyes for training our own model that will predict which direction we are looking. We'll use the Keras library to build a supervised (labeled) model, and some considerations should be taken into account right about now. When Keras loads images for training a suprevised model, it expects a file structure for training and test data resembeling the following : 

    ├── main.py
    ├── data
    │   ├── test
    │   │   ├── class1
    │   │   │   ├── img4.png
    │   │   │   └── img5.png
    │   │   └── class2
    │   │       ├── img6.png
    │   │       └── img7.png
    │   └── train
    │       ├── class1
    │       │   ├── img0.png
    │       │   └── img1.png
    │       └── class2
    │           ├── img2.png
    │           └── img3.png

This way it knows the labels (as directory name) for each image. So what we need to do now is build that directory structure and start saving images to directories
For now, lets make the `training` and `test` directories and create classes of `left` and `right`. Here's an example of what I have : 

    ├── images
    │   ├── test
    │   │   ├── left
    │   │   └── right
    │   └── train
    │       ├── left
    │       └── right
    ├── main.py


The obvious question now is how to direct our saved images of eyes to each of the directories. It is common in machine learning to use an 80/20 split between training and testing data. There are a number of ways to hack together such split, but here's the approach I'm taking.

Algorithm

    Detect faces
    for each face:
        detect all eyeballs
        for each eyeball:
            randomly select 'train' or 'test' with 0.8 and 0.2 probability respectively. 
            save image of eyeball roi to selected train or test directory under appropriate class

For the sake of pythonic modularity, lets put the random selection and saving into a separate fuction, `selective_save()`. We can do the same with the code we have up to now, let's call it `data_collect()` and call it within the main block. Here's what we have for code at this point. 

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
