from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.label import Label
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import urllib.request
from bs4 import BeautifulSoup
import requests

class CamApp(App):

    def build(self):
        self.img1=Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)

        try:
            self.augDics = self.lod(self.rd_url('https://companycrm.ir/markeroverlays/index.php', ''))  # online
        except:
            self.augDics = self.loadAugImages("images") #offline

        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        #cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, dt):
        # display image from cam in opencv window
        ret, frame = self.capture.read()

        arucoFound = self.findArucoMarkers(frame)

        if len(arucoFound[0]) != 0:

            if self.augDics[int(arucoFound[1])][1] == "mp4":
                _, image_video = self.augDics[int(arucoFound[1])][0].read()
            else:
                image_video = self.augDics[int(arucoFound[1])][0]

            frame = self.augmentAruco(arucoFound[0], arucoFound[1], frame,image_video)

        # cv2.imshow("CV2 Image", frame)
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer.
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1


    def loadAugImages(self, path):

        myList = os.listdir(path)
        noOfMarkers = len(myList)
        print("Total # of markers:", noOfMarkers)

        audDics = {}
        for imgpath in myList:
            key = int(os.path.splitext(imgpath)[0])
            if os.path.splitext(imgpath)[-1] == ".jpg" or os.path.splitext(imgpath)[-1] == ".png" :
                imgAug = [cv2.imread(f'{path}/{imgpath}'), "jpg"]
            else:
                imgAug = [cv2.VideoCapture(f'{path}/{imgpath}'), "mp4"]

            # print(imgAug)
            audDics[key] = imgAug
        return audDics


    def findArucoMarkers(self ,img, markerSize=6, totalMarkers=250, draw=True):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
        arucoDict = aruco.Dictionary_get(key)
        arucoParam = aruco.DetectorParameters_create()

        bbox, ids, rejected = aruco.detectMarkers(image=imgGray, dictionary=arucoDict, parameters=arucoParam)

        # print(ids)
        if draw:
            aruco.drawDetectedMarkers(img, bbox)

        return [bbox, ids]

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                # cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours

    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)

    def augmentAruco(self ,bbox, id , img ,imgAgu , drawId=True):

        tl = bbox[0][0][0][0], bbox[0][0][0][1]
        tr = bbox[0][0][1][0], bbox[0][0][1][1]
        br = bbox[0][0][2][0], bbox[0][0][2][1]
        bl = bbox[0][0][3][0], bbox[0][0][3][1]

        h, w, c = imgAgu.shape

        pts1 = np.array([tl, tr, br, bl])
        pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        matrix, _ = cv2.findHomography(pts2, pts1)

        imgOut = cv2.warpPerspective(imgAgu, matrix, (img.shape[1], img.shape[0]))

        cv2.fillConvexPoly(img, pts1.astype(int), (0,0,0))
        #imgOut = img + imgOut

        # می خوای تصویر روی گراف نشون داده بشه این خروجی رو فعال کن
        #return imgOut
        # اگه می خوای تصویر که پخش می شه تمام صفحه باشه این خروجی رو فعال کن
        return imgAgu


    def rd_url(self,url,ext):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        page = requests.get(url,headers=headers).text
        soup = BeautifulSoup(page, 'html.parser')
        return [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

    def lod(self,path):
        noOfMarkers = len(path)
        print("Total # of markers:", noOfMarkers)
        audDics = {}
        for imgpath in path:
            x = imgpath.split("/")[-1]
            key = int(x.split(".")[0])
            if x.split(".")[-1] == "jpg":
                req = urllib.request.urlopen(imgpath)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, 1)  # 'Load it as it is'
                imgAug = [img, "jpg"]
            else:
                cap = cv2.VideoCapture(imgpath)
                imgAug = [cap, "mp4"]
            audDics[key] = imgAug
        return audDics


if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()

