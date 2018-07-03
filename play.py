import cv2
import sys
import time
from PIL import Image


'''
points=[]
print(points)
rect=(2,3,4,3)   # rect = x,y,w,h
x=int((2*rect[0]+rect[2])/2)
print(x)
y=int((2*rect[1]+rect[3])/2)
print(y)
points.append((x,y))
points.append((7,8))

print(points)
del(points[0])
print(points)
'''





(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(minor_ver)

if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]
    print(tracker_type)

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()



    # Read video
    #video = cv2.VideoCapture("traffic_4.mp4")
    video = cv2.VideoCapture("/home/aiuser/Downloads/hik_doctors_1080.mp4")
    video = cv2.VideoCapture("/home/aiuser/Downloads/Прогулка по офису компании Nival.mp4")
    print('video')

    # Exit if video not opened.
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()

    # skip first n frames
    for i in range(780):
        ok, frame = video.read()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print
        'Cannot read video file'
        sys.exit()
    print(ok)

    imgF = Image.fromarray(frame)
    im_width, im_height = imgF.size
    print("{} {} {} {}".format(" width -> ", im_width, " height -> ", im_height))

    # Define an initial bounding box
    #bbox = (287, 23, 86, 320) # x, y, w, h

    # унификация размера выходящего изображения
    #frame = cv2.resize(frame, (720, 480))

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # унификация размера выходящего изображения
        #frame = cv2.resize(frame, (720, 480))
        #time.sleep(0.1)  # трекеры работают слишком быстро - замедление

        # Update tracker
        ok, bbox = tracker.update(frame)
        print(bbox)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break


'''
print("Координаты точки A(x1;y1):")
x1 = 0.0
y1 = 4.0

print("Координаты точки B(x2;y2):")
x2 = 6.0
y2 = 0.0

print("Уравнение прямой, проходящей через эти точки:")
k = (y1 - y2) / (x1 - x2)
b = y2 - k * x2
print(" y = %.2f*x + %.2f" % (k, b))

x=3
y = -0.67*x + 4.00
print(y)
'''

'''
x= 6
y=5
dist = -0.67*x + 4.00 - y
print(dist)
'''

'''
Задача о принадлежности точки многоугольнику
В основе алгоритма лежит идея подсчёта количества пересечений луча, исходящего из данной точки в направлении 
горизонтальной оси, со сторонами многоугольника. Если оно чётное, точка не принадлежит многоугольнику. 
В данном алгоритме луч направлен влево.

def inPolygon(x, y, xp, yp):
    c = 0
    for i in range(len(xp)):
        if ((yp[i] <= y and y < yp[i - 1]) or (yp[i - 1] <= y and y < yp[i])) and (
                x > (xp[i - 1] - xp[i]) * (y - yp[i]) / (yp[i - 1] - yp[i]) + xp[i]): c = 1 - c
    return c


print(inPolygon(3, 1, (0, 0, 6), (0, 4, 0)))
'''

'''
Решение задачи нахождения точки относительно прямой.
Точка находится под прямой
Пусть, имеется точка A(x, y), где x, y - нецелые числа. Уравнение прямой y = kx + b. Функция, возвращающая y в зависимости от x:

def getY(x, k, b):
    return k*x+b
    
    
Узнать ниже ли точка прямой или нет, можно, подставив ее x в уравнение прямой:

if y < getY(x, y, k):
    print("Точка под прямой.")
    
    
Точка на прямой
Пусть, имеется точка A(x, y), где x, y - нецелые числа. Узнать на прямой ли точка или нет, можно, подставив ее x в уравнение прямой:

if y == getY(x, y, k):
    print("Точка на прямой.")
    
    
Точка над прямой
Пусть, имеется точка A(x, y), где x, y - нецелые числа. Узнать выше прямой ли точка или нет, можно, подставив ее x в уравнение прямой:

if y > getY(x, y, k):
    print("Точка над прямой.")
'''