import collections
from typing import List, Any

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
from vehicle import vehicle
import PIL.Image as Image
from collections import defaultdict
from io import StringIO
from PIL import Image
import time
from multiprocessing.pool import ThreadPool
import threading
import time
# import openalpr_api
# from openalpr_api.rest import ApiException
import numpy as np
import cv2
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import PIL.ImageDraw as ImageDraw
import json
import re

# This is needed to display the images.
from utils import label_map_util
from utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

'''
api = openalpr_api.DefaultApi()
secret_key = ''
country = 'us'
recognize_vehicle = 0
state = ''
return_image = 0
topn = 1
prewarp = ''

def getLicensePlateNumber(filer):
	try:
		js = api.recognize_file(filer, secret_key, country, recognize_vehicle=recognize_vehicle, state=state, return_image=return_image, topn=topn, prewarp=prewarp)

		js=js.to_dict()
		#js=list(str(js))
		X1=js['results'][0]['coordinates'][0]['x']
		Y1=js['results'][0]['coordinates'][0]['y']
		X2=js['results'][0]['coordinates'][2]['x']
		Y2=js['results'][0]['coordinates'][2]['y']
		img=cv2.imread(filer)
		rimg=img[Y1:Y2,X1:X2]
		frame3=rimg
		img3 = Image.fromarray(frame3)
		w,h=img3.size
		asprto=w/h
		frame3=cv2.resize(frame3,(150,int(150/asprto)))
		cv2image3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
		img3 = Image.fromarray(cv2image3)
		imgtk3 = ImageTk.PhotoImage(image=img3)
		display4.imgtk = imgtk3 #Shows frame for display 1
		display4.configure(image=imgtk3)
		display5.configure(text=js['results'][0]['plate'])
	except ApiException as e:
	    print("Exception: \n", e)
'''

trackers = []
vehicles = []
count = 0


def matchVehicles(currentFrameVehicles, im_width, im_height, image):
    # обьявление этой функции предназначено для унификации моего подхода(currentFrameVehicles (box)), 
    # с старым подходом (currentFrameVehicles(box+colors)). Сделано это потому, что currentFrameVehicles(box+colors) 
    # передаёт дополнительный параметр color, кроме этого координаты передаются не нормализированные и в не подходящем 
    # формате
    def do_work_if_vehicles_empty(x, y, w, h):
        # координаты центроида
        X = int((x + x + w) / 2)
        Y = int((y + y + h) / 2)
        # условие пересечения последней верхней линии, если машина не пересекла линию:
        if Y > yl5:
            # cv2.circle(image,(X,Y),2,(0,255,0),4)
            # print('Y=',Y,'  y1=',yl1)
            vehicles.append(vehicle((x, y, w, h)))

    def do_work_if_vehicles_full(x, y, w, h):
        index = 0
        # растояние между points используется для проверки на сопоставление: тот-же автомобиль отслеживается или
        # это другой автомобиль, используются: distance, и diagonal; в классе vehicles для этого
        # используют методы: updatePosition, setCurrentFrameMatch, getTracking, getNext... и тд.
        ldistance = 999999999999999999999999.9
        X = int((x + x + w) / 2)
        Y = int((y + y + h) / 2)
        if Y > yl5:
            # print('Y=',Y,'  y1=',yl1)
            # cv2.circle(image,(X,Y),4,(0,0,255),8)
            # проверяем какая точка из боксов соответствует какому обьекту vehicle
            for i in range(len(vehicles)):
                if vehicles[i].getTracking() == True:
                    # print(vehicles[i].getNext(),i)
                    # distance разница между предсказанным центром обьекта, и фактическими координатами центра бокса,
                    # если растояние между ними меньше чем диагональ обьекта, значит новая точка относится к текущему
                    # обьекту, если нет, то точка относится к другому обьекту (к другой машине)
                    distance = ((X - vehicles[i].getNext()[0]) ** 2 + (Y - vehicles[i].getNext()[1]) ** 2) ** 0.5

                    if distance < ldistance:
                        ldistance = distance
                        index = i

            diagonal = vehicles[index].diagonal

            # если растояние между точками меньше чем диагональ бокса, для обьекта добавляем новый центр и новую диагональ
            if ldistance < diagonal:
                vehicles[index].updatePosition((x, y, w, h))
                vehicles[index].setCurrentFrameMatch(True)
            # если нет, создаём новый обьект класса vehicle и добавляем его в лист vehicles
            else:
                # blue for last position
                # cv2.circle(image,tuple(vehicles[index].points[-1]),2,(255,0,0),4)
                # red for predicted point
                # cv2.circle(image,tuple(vehicles[index].getNext()),2,(0,0,255),2)
                # green for test point
                # cv2.circle(image,(X,Y),2,(0,255,0),4)

                # cv2.imshow('culprit',image)
                # time.sleep(5)
                # print(diagonal,'               ',ldistance)
                vehicles.append(vehicle((x, y, w, h)))

    # если список машин пустой
    if len(vehicles) == 0:
        # проверка на адекватность currentFrameVehicles подходу с использованием трэкеров или старому.
        if not isinstance(currentFrameVehicles, list):
            for box, color in currentFrameVehicles:
                (y1, x1, y2, x2) = box
                # пересчитываем координаты боксов на координаты боксов на фрейме при этом пересчитываем х, у, высота,ширина
                (x, y, w, h) = (
                    x1 * im_width, y1 * im_height, x2 * im_width - x1 * im_width, y2 * im_height - y1 * im_height)
                do_work_if_vehicles_empty((x, y, w, h))
        # при использовании трэкера координаты идут нормализированные в нужном нам формате
        else:
            for box in currentFrameVehicles:
                (x, y, w, h) = box
                do_work_if_vehicles_empty(x, y, w, h)

    # если в vehicles есть обьекты, для каждого обьекта vehicles:
    else:
        for i in range(len(vehicles)):
            vehicles[i].setCurrentFrameMatch(False)
            vehicles[i].predictNext()  # предсказание где будет следующий центр обьекта
        for i in range(len(vehicles)):
            if vehicles[i].getCurrentFrameMatch() == False:
                vehicles[i].increaseFrameNotFound()
            if not isinstance(currentFrameVehicles, list):
                for box, color in currentFrameVehicles:
                    (y1, x1, y2, x2) = box
                    # пересчитываем координаты боксов на координаты боксов на фрейме при этом пересчитываем х, у, высота,ширина
                    (x, y, w, h) = (
                        x1 * im_width, y1 * im_height, x2 * im_width - x1 * im_width, y2 * im_height - y1 * im_height)
                    do_work_if_vehicles_full(x, y, w, h)
            # при использовании трэкера координаты идут нормализированные в нужном нам формате
            else:
                for box in currentFrameVehicles:
                    (x, y, w, h) = box
                    do_work_if_vehicles_full(x, y, w, h)


# print(len(vehicles))
# pool = ThreadPool(processes=1)


def checkRedLightCrossed(img):
    global count
    for v in vehicles:
        if v.crossed == False and len(v.points) >= 2:
            x1, y1 = v.points[0]
            x2, y2 = v.points[-1]
            if y1 > yl3 and y2 < yl3:
                count += 1
                v.crossed = True
                bimg = img[int(v.rect[1]):int(v.rect[1] + v.rect[3]), int(v.rect[0]):int(v.rect[0] + v.rect[2])]
                frame2 = bimg
                img2 = Image.fromarray(frame2)
                w, h = img2.size
                asprto = w / h
                frame2 = cv2.resize(frame2, (250, int(250 / asprto)))
                cv2.imshow('BROKE', bimg)
                #cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                '''
                img2 = Image.fromarray(cv2image2)
                imgtk2 = ImageTk.PhotoImage(image=img2)
                display2.imgtk = imgtk2  # Shows frame for display 1
                display2.configure(image=imgtk2)
                '''
                # cv2.imshow('BROKE',bimg)
                name = 'Rule Breakers/culprit' + str(time.time()) + '.jpg'
                cv2.imwrite(name, bimg)

                '''
                tstop = threading.Event()
                thread = threading.Thread(target=getLicensePlateNumber, args=(name,))
                thread.daemon = True
                thread.start()
                '''
            # cv2.imwrite('culprit.png',bimg)


# display3.configure(text=count)


def checkRegionCrossed(img):
    global count
    for v in vehicles:
        if v.crossed == False and len(v.points) >= 2:
            x1, y1 = v.points[0]
            x2, y2 = v.points[-1]
            if y1 > yl3 and y2 < yl3:
                count += 1
                v.crossed = True
                bimg = img[int(v.rect[1]):int(v.rect[1] + v.rect[3]), int(v.rect[0]):int(v.rect[0] + v.rect[2])]
                frame2 = bimg
                img2 = Image.fromarray(frame2)
                w, h = img2.size
                asprto = w / h
                frame2 = cv2.resize(frame2, (250, int(250 / asprto)))
                cv2.imshow('Alarm', frame2)

                name = 'Rule Breakers/culprit' + str(time.time()) + '.jpg'
                cv2.imwrite(name, bimg)

                '''
                tstop = threading.Event()
                thread = threading.Thread(target=getLicensePlateNumber, args=(name,))
                thread.daemon = True
                thread.start()
                '''
            # cv2.imwrite('culprit.png',bimg)


# display3.configure(text=count)


def checkSpeed(ftime, img):
    for v in vehicles:
        if v.speedChecked == False and len(v.points) >= 2:
            x1, y1 = v.points[0]
            x2, y2 = v.points[-1]
            if y2 < yl1 and y2 > yl3 and v.entered == False:
                v.enterTime = ftime
                v.entered = True
            elif y2 < yl3 and y2 > yl5 and v.exited == False:
                v.exitTime = ftime
                v.exited == False
                v.speedChecked = True
                speed = 60 / (v.exitTime - v.enterTime)
                print(speed)
                bimg = img[int(v.rect[1]):int(v.rect[1] + v.rect[3]), int(v.rect[0]):int(v.rect[0] + v.rect[2])]
                frame2 = bimg
                img2 = Image.fromarray(frame2)
                w, h = img2.size
                asprto = w / h
                frame2 = cv2.resize(frame2, (250, int(250 / asprto)))
                cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                img2 = Image.fromarray(cv2image2)
                imgtk2 = ImageTk.PhotoImage(image=img2)
                display2.imgtk = imgtk2  # Shows frame for display 1
                display2.configure(image=imgtk2)
                display3.configure(text=str(speed)[:5] + 'Km/hr')
                if speed > 60:
                    # cv2.imshow('BROKE',bimg)
                    name = 'Rule Breakers/culprit' + str(time.time()) + '.jpg'
                    cv2.imwrite(name, bimg)

                    '''
                    tstop = threading.Event()
                    thread = threading.Thread(target=getLicensePlateNumber, args=(name,))
                    thread.daemon = True
                    thread.start()
                    '''


'''
# Эта часть записывает видео в отдельный файл, уменьшая размер до 640/480 на данный момент эта опция не нужна
filename = "testoutput.avi"
codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # fourcc stands for four character code
framerate = 10
resolution = (640, 480)
VideoFileOutput = cv2.VideoWriter(filename, codec, framerate, resolution)

# WebcamVideoStream
vs = WebcamVideoStream(src='Set01_video01.mp4').start()
'''

cap = cv2.VideoCapture('/home/aiuser/Downloads/traffic_01.avi')  # 0 stands for very first webcam attach
#cap = cv2.VideoCapture('traffic_4.mp4')  # 0 stands for very first webcam attach

for i in range(20):
    ret, imgF = cap.read(0)
imgF = Image.fromarray(imgF)
im_width, im_height = imgF.size
xl1 = 0
xl2 = im_width - 1
yl1 = im_height * 0.5
yl2 = yl1
ml1 = (yl2 - yl1) / (xl2 - xl1)
intcptl1 = yl1 - ml1 * xl1

xl3 = 0
xl4 = im_width - 1
yl3 = im_height * 0.25
yl4 = yl3
ml2 = (yl4 - yl3) / (xl4 - xl3)
intcptl2 = yl3 - ml2 * xl3

xl5 = 0
xl6 = im_width - 1
yl5 = im_height * 0.1
yl6 = yl5
ml3 = (yl6 - yl5) / (xl6 - xl5)
intcptl3 = yl5 - ml3 * xl5
ret = True
start = time.time()
c = 0

# Инициализация модели
# What model to download.
MODEL_NAME = 'Cars'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/output_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('Cars', 'car_label_map.pbtxt')
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

sesser = tf.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

'''
# Tkinter — кросс-платформенная графическая библиотека на основе средств Tk.
window = tk.Tk()  # Makes main window
window.wm_title("Monitor")
window.columnconfigure(0, {'minsize': 1020})
window.columnconfigure(1, {'minsize': 335})
frame = tk.Frame(window)
frame.grid(row=0, column=0, rowspan=5, sticky='N', pady=10)
frame2 = tk.Frame(window)
frame2.grid(row=0, column=1)
frame3 = tk.Frame(window)
frame3.grid(row=1, column=1)
frame4 = tk.Frame(window)
frame4.grid(row=2, column=1)
frame5 = tk.Frame(window)
frame5.grid(row=3, column=1)
frame2.rowconfigure(1, {'minsize': 250})
frame3.rowconfigure(1, {'minsize': 80})
frame4.rowconfigure(1, {'minsize': 150})
frame5.rowconfigure(1, {'minsize': 80})
'''


def main(sess=sesser):
    ''' global masterframe
	global started '''

    global trackers
    global vehicles

    while True:
        # if True:
        # fTime = time.time()
        _, image_np = cap.read(0)
        print(cap.get(1))
        # image_np = imutils.resize(image_np, width=400)
        # Definite input and output Tensors for detection_graph
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        # t1 = time.time()
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # t2 = time.time()
        # print(t2-t1)
        print("{} {}".format("vehicles -> ", len(vehicles)))
        print("{} {}".format("trackers -> ", len(trackers)))
        # После того как модель выдала нам новые боксы очищаем список трэкеров
        trackers.clear()
        #print(len(trackers))
        # Удаляем устаревшие обьекты из vehicles
        # вот это безумие(с введением временного списка) мне пришлось сделать с целью удаления из vehicles тех
        # обьектов которые не отслеживаются (getTracking()=False), для этого я создал временный список в который
        # итеративно занёс обьекты отслеживаемые, и перезаписал список vehicles.
        if cap.get(1) % 5 == 0:
            temp_vehicles_list = []
            for v in vehicles:
                if v.getTracking() == True:
                    temp_vehicles_list.append(v)
            vehicles = temp_vehicles_list

        # Visualization of the results of a detection.
        # img = image_np
        '''
        imgF, bbox_track_list = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)
        '''
        # моя инновация часть первая: после того как модель передаст нам боксы, мы создаём для каждого бокса трэкеры
        image_np, trackers = getTrackers(image_np,
                                         np.squeeze(boxes),
                                         np.squeeze(classes).astype(np.int32),
                                         np.squeeze(scores),
                                         category_index,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=.5,
                                         )

        # моя инновация часть вторая: для последующих n кадров, извлекаем боксы из трекеров в список bbox_track_list
        # рисуем на фрэйме боксы и передаём bbox_track_list для последующей обработки.
        for i in range(5):
            bbox_track_list = []
            _, image_np = cap.read(0)
            for j in range(len(trackers)):
                ok, bbox = trackers[j].update(image_np)
                if ok:
                    bbox_track_list.append(bbox)
                '''
                else:
                    trackers[j].release()
                '''
                # рисуем на фрэйме бокс
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(image_np, p1, p2, (255, 0, 0), 2, 1)

            # print("{} {}".format("bbox_track_list -> ", bbox_track_list))
            # print("{} {}".format("vehicles -> ", len(vehicles)))
            # print("{} {}".format("trackers -> ", len(trackers)))

            matchVehicles(bbox_track_list, im_width, im_height, image_np)
            #checkRegionCrossed(img)
            # checkRedLightCrossed(imgF)
            # checkSpeed(fTime, img)

            # для каждого из обьектов(машина), после того как информация о нём была обновлена (matchVehicles), рисуются все его
            # поинты (точки центроидов).
            for v in vehicles:
                if v.getTracking() == True:
                    for p in v.getPoints():
                        cv2.circle(image_np, p, 3, (200, 150, 75), 6)

                # print(ymin*im_height,xmin*im_width,ymax*im_height,xmax*im_width)
                # cv2.rectangle(image_np,(int(xmin*im_width),int(ymin*im_height)),(int(xmax*im_width),int(ymax*im_height)),(255,0,0),2)
            cv2.line(image_np, (int(xl1), int(yl1)), (int(xl2), int(yl2)), (0, 255, 0), 3)
            cv2.line(image_np, (int(xl3), int(yl3)), (int(xl4), int(yl4)), (0, 0, 255), 3)
            cv2.line(image_np, (int(xl5), int(yl5)), (int(xl6), int(yl6)), (255, 0, 0), 3)
            # VideoFileOutput.write(image_np)  # Запускает запись видео в файл, на данный момент эта опция не нужна
            # Ресайз фрейма под габариты визуализации, и визуализация.
            frame = cv2.resize(image_np, (720, 480))
            k = cv2.waitKey(1) & 0xff
            if k == 27: break
            cv2.imshow("Tracker", frame)

            # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            # img = Image.fromarray(cv2image)
            # imgtk = ImageTk.PhotoImage(image=img)
            # display1.imgtk = imgtk  # Shows frame for display 1
            # display1.configure(image=imgtk)
            # print('count -> '), print(count)
            # print('vehicles -> '), print(len(vehicles))

    # window.after(1, main)

'''
# Tkinter — кросс-платформенная графическая библиотека на основе средств Tk.
lbl1 = tk.Label(frame, text='Vehicle Detection And Tracking', font="verdana 12 bold")
lbl1.pack(side='top')
lbl2 = tk.Label(frame2, text='Vehicle Breaking Traffic Rule', font="verdana 10 bold")
lbl2.grid(row=0, column=0, sticky='S', pady=10)
lbl3 = tk.Label(frame3, text='Veicle Speed', font="verdana 10 bold")
lbl3.grid(row=0, column=0, sticky='S', pady=10)
lbl4 = tk.Label(frame4, text='Detected License Plate', font="verdana 10 bold")
lbl4.grid(row=0, column=0)
lbl5 = tk.Label(frame5, text='Extracted License Plate Number', font="verdana 10 bold")
lbl5.grid(row=0, column=0)
display1 = tk.Label(frame)
display1.pack(side='bottom')  # Display 1
display2 = tk.Label(frame2)
display2.grid(row=1, column=0)  # Display 2
display3 = tk.Label(frame3, text="", font="verdana 14 bold", fg='red')
display3.grid(row=1, column=0)
display4 = tk.Label(frame4)
display4.grid(row=1, column=0)
display5 = tk.Label(frame5, text="", font="verdana 24 bold", fg='green')
display5.grid(row=1, column=0)
'''

masterframe = None
started = False


def getTrackers(image,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=False,
                max_boxes_to_draw=20,
                min_score_thresh=.5,
                agnostic_mode=False,
                ):

    global trackers

    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    # для каждого бокаса полученного от модели (или max_boxes_to_draw), при условии что scores бокса больше чем
    # min_score_thresh: переводим координаты бокса из относительных в нормализированные, при этом задаём не нижнюю
    # точку, а длину и ширину (что требуется для трекеров)
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            (y1, x1, y2, x2) = box
            (x, y, w, h) = (
                x1 * im_width, y1 * im_height, x2 * im_width - x1 * im_width, y2 * im_height - y1 * im_height)

            # создаём и инициализируем трэкер
            # test two different trackers KCF and MedianFlow. при тесте GOTURN, обязательно добавь goturn.caffemodel
            # и goturn.prototxt из папки /home/aiuser/Andreas_Study/vehicle_counting/goturn-files-master
            tracker = cv2.TrackerKCF_create()
            #tracker = cv2.TrackerMedianFlow_create()
            #tracker = cv2.TrackerGOTURN_create()
            tracker.init(image, (x, y, w, h))

            # рисуем на фрэйме бокс

            p1 = (int(x), int(y))
            p2 = (int(x+w), int(y+h))
            cv2.rectangle(image, p1, p2, (255, 0, 0), 2, 1)



            # добавляем трэкер в лист трэкеров
            trackers.append(tracker)

    return image, trackers  # возможно, нет смысла возвращать лист trackers так как нам не надо его возвращать: это
    # глобальная переменная, если мы вынесем эту функцию в отдельный файл, тогда стоит сделать его возвращамемым..


def stream():
    global masterframe
    global started
    global c
    global tim
    cap = cv2.VideoCapture('video7.mp4')
    while True:
        started, masterframe = cap.read()
        time.sleep(0.034)


'''thread = threading.Thread(target=stream)
thread.daemon = True
thread.start()'''


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        sesser = sess
        main(sess)  # Display
# window.mainloop()  # Starts GUI
