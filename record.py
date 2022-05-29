from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


class func:

    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("face Recognition System for Attendance")

        title_lbl = Label(self.root, text="FACE RECOGNITION ATTENDANCE PAGE", font=("times new roman", 40, "bold"), bg="white",
                          fg="dark blue")
        title_lbl.place(x=0, y=0, width=1530, height=50)

        # 1st image
        img_top = Image.open(r"ImagesAttendance\mid.jpg")
        img_top = img_top.resize((650, 700), Image.ANTIALIAS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)

        f_lbl = Label(self.root, image=self.photoimg_top)
        f_lbl.place(x=0, y=60, width=650, height=700)

        # 2nd image
        img_bottom = Image.open(r"ImagesAttendance\mid.jpg")
        img_bottom = img_bottom.resize((950, 700), Image.ANTIALIAS)
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)

        f_lbl = Label(self.root, image=self.photoimg_bottom)
        f_lbl.place(x=650, y=60, width=950, height=700)

        # button

        b1_1 = Button(f_lbl, text="Click to open Camera Here", command=self.face_, cursor="hand2",
                      font=("times new roman", 20, "bold"), bg="purple", fg="white")
        b1_1.place(x=0, y=80, width=500, height=80)

    def face_(self):

        path = r"SampleImages"

        images = []

        classNames = []

        myList = os.listdir(path)
        print(myList)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        print(classNames)

        def findEncodings(images):

            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, _ = img.shape
                # location is in css order - top, right, bottom, left
                face_location = (0, width, height, 0)

                encode = face_recognition.face_encodings(img)[0]

                encodeList.append(encode)
            return encodeList

        def markAttendence(name):

            with open('Attendance.csv', 'r+') as f:

                myDataList = f.readlines()
                nameList = []
                print(myDataList)
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                if name not in nameList:
                    now = datetime.now()
                    dtString = now.strftime('%H:%M"%S')
                    f.writelines(f'\n{name},{dtString}')

        encodeListKnown = findEncodings(images)
        print('encoding complete')

        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        while True:

            success, img = cap.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):

                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                # print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    # print(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendence(name)

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == 13:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    obj = func(root)
    root.mainloop()