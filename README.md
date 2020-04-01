# LookyLoo
_A project in real-time image recognition and classification._

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
When starting a new project, I like to keep version control top of mind. There will be many iterations to this project as it develops, making version control a critical component to this project. Let’s start a new (empty) repo on GitHub, clone it to my local machine, and make the first branch. 

`$ git clone <my_project_repo>.git`
`$ git branch starting-out`


After making an empty file called `main.py`, I’ll open up a terminal and spin up a virtual python environment specifically for this project. I like working with Python3.6+ so lets get it running and start installing the packages we KNOW we need immediately. 


## Data Acquisition v0.0
Lets 

