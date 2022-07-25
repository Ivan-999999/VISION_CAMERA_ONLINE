#---ПРОГРАММА ДЛЯ РАСПОЗНАВАНИЯ ОБЪЕКТОВ В РЕЖИМЕ РЕАЛЬНОГО ВРЕМЕНИ.
import cv2
from imageai.Detection import ObjectDetection
import time


camera = cv2.VideoCapture(0)                           #---Захват видеопотока. Как аргумент - идентификатор устройства, 0 - камера ноутбука. Если подключена другая - попробовать другой идентификатор. Либо указываем путь до файла.

#---Загрузка модели делается только один раз, поэтому данная часть кода вне цикла while.
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()                     # ---"yolo" - модель нейронной сети, которую будем использовать. v3 - версия 3 признана самой быстрой на данный момент.
detector.setModelPath('resnet50_coco_best_v2.1.0.h5')  # ---Указываем путь до файла с нейронной сетью (просто его название)
detector.loadModel()

finish = 0
arrayDetection = []

while camera.isOpened():                                                          #---Создаем цикл, работающий пока не выключится камера, или не закончится видео.
    ret, frame = camera.read()                                                    #---Считываем фрейм с нашей камеры.

    start = time.time()
    if start - finish > 0.5:                                                      #---Фиксация каждые пол секунды.
        _, arrayDetection = detector.detectObjectsFromImage(input_image=frame, input_type="array", output_type="array")  # ---Какое изображение обрабатываем и какое имя будет у него после обработки. В input_image мы можем передавать не только изображения, но и frame, и работать с видеопотоком.
        finish = time.time()
        print(arrayDetection)

    for obj in arrayDetection:
        coord = obj["box_points"]
        cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255))
        cv2.putText(frame, obj["name"], (coord[0], coord[1] - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


    cv2.imshow('Test', frame)#---Нужно отобразить изображение с камеры в каком то окошке.
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

















# #---ЧАСТЬ КОДА ДЛЯ РАБОТЫ С КАМЕРОЙ.
# camera = cv2.VideoCapture(0)   #---Захват видеопотока. Как аргумент - идентификатор устройства, 0 - камера ноутбука. Если подключена другая - попробовать другой идентификатор.
# while camera.isOpened():       #---Создаем цикл, работающий пока не выключится камера, или не закончится видео.
#     ret, frame = camera.read() #---Считываем фрейм с нашей камеры.
#     cv2.imshow('Test', frame)#---Нужно отобразить изображение с камеры в каком то окошке.
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
# camera.close()
# cv2.destroyAllWindows()


# #---ЧАСТЬ КОДА ДЛЯ РАСПОЗНАВАНИЯ ОБЪЕКТОВ НА ИЗОБРАЖЕНИИ.
# detector = ObjectDetection()
# detector.setModelTypeAsTinyYOLOv3()   #---"yolo" - модель нейронной сети, которую будем использовать. v3 - версия 3 признана самой быстрой на данный момент.
# detector.setModelPath('yolo-tiny.h5') #---Указываем путь до файла с нейронной сетью (просто его название)
# detector.loadModel()                  #---Загружаем модель в оперативную память (если модель весит 200 мб, она займет именно столько оперативной памяти.
#
# x = detector.detectObjectsFromImage(input_image="utExDsJH6IA.jpg", output_image_path="new_utExDsJH6IA.jpg") #---Какое изображение обрабатываем и какое имя будет у него после обработки.
# print(x)







