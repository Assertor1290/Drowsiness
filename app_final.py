# Libraries for ml and heart
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import base64
import pandas as pd
import webbrowser

# Libraries for ear
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
from threading import Event
import dlib
import cv2

# Libraries for plotting
import numpy as np
import serial
#import time
import matplotlib.pyplot as plt
from collections import deque

import matplotlib.dates as mdates
import datetime
from playsound import playsound

# code for plotting
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

max_samples = 100
#max_x = max_samples
max_x = 200
max_rand = 100

# ml model code
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

pickle_in_2 = open('scaler.sav', 'rb')
norm = pickle.load(pickle_in_2)

# EAR code
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="alarm.wav", help="path alarm .WAV file")
args = ap.parse_known_args()[0]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../../68 face landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# arduino read code
ser = serial.Serial('COM3', 9600, timeout=1)
#time.sleep(2)
Event().wait(2.0)

#@st.cache()
def sound_alarm(path):
    playsound('alarm.wav')
    
def get_base64_of_bin_file(bin_file):
    """function to read png file"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    """function to display png as bg"""
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(bpm_input, ear_input):   
 
    X_train_norm = norm.transform([[bpm_input, ear_input]])
    
    # Making predictions 
    prediction = classifier.predict( 
        X_train_norm)
     
    if prediction == 0:
        pred = 'Drowsy'
    else:
        pred = 'Awake'
    return pred
      
    
def design():       
    set_png_as_page_bg('733911.jpg')
    # front end elements of the web page 
    html_temp = """
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Driver Drowsiness Detection App</h1> 
    </div>
    """  
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    success_style = """
    <style>
    p {
        color:#000000;
        font-size:18px;
        font-weight: bold;
        border-radius:2%;
    }

    </style>
    """
    st.markdown(success_style,unsafe_allow_html=True)
    

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def main():
    design()
    COUNTER = 0
    ALARM_ON = False
    
    #ser = serial.Serial('COM3', 9600, timeout=1)
    vs = VideoStream(src=args.webcam).start()
    Event().wait(1)
    
    #x = np.arange(0, 200, 2)
    y = deque(np.zeros(max_samples), max_samples)
    x = [datetime.datetime.now() + datetime.timedelta(seconds=i) for i in range(len(y))]
    z = deque(np.zeros(max_samples), max_samples)
    
    ax1.set_ylim(0, 0.5)
    linedraw, = ax1.plot(x, np.array(y))
    the_plot = st.pyplot(fig1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    
    for label in ax1.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(45)
    
    ax2.set_ylim(0, 140)
    linedraw2, = ax2.plot(x, np.array(z))
    the_plot2 = st.pyplot(fig2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    
    for label in ax2.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(45)
    
    fig1.suptitle('EAR', fontsize=20)
    fig1.text(0.04, 0.5,'EAR values', va='center', rotation='vertical')
    fig2.suptitle('BPM', fontsize=20)
    fig2.text(0.04, 0.5,'BPM values', va='center', rotation='vertical')
    
    while True:
        # to read values simulataneously per second
        Event().wait(1)
        frame = vs.read()
        line = ser.readline()
        
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            def animate_ear(ear): 
                linedraw.set_ydata(np.array(y))
                the_plot.pyplot(fig1)
                y.append(ear)
                
            def animate_bpm(bpm): 
                linedraw2.set_ydata(np.array(z))
                the_plot2.pyplot(fig2)
                z.append(bpm)

            if line:
                string = line.decode() 
                num = int(string)      
                print('BPM:{}, EAR:{}'.format(num, ear))
                result = ""
                result = prediction(num, ear)
                print(result)
                
                if result=='Drowsy':
                    COUNTER+=1
                    if COUNTER >= 5:
                        if not ALARM_ON:
                            ALARM_ON = True
                            COUNTER = 0

                            if args.alarm != "":
                                t = Thread(target=sound_alarm, args=(args.alarm,))
                                t.deamon = True
                                t.start()
                                ALARM_ON = False
                
                else:
                    COUNTER = 0
                    ALARM_ON = False
                
                animate_ear(ear)
                animate_bpm(num)
                Event().wait(0.1)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()
    ser.close()
    

if __name__=='__main__': 
    main()

