import gpiod
import time
import cv2

# Пути к файлам модели
model_path = "MobileNetSSD_deploy.caffemodel"  # укажите путь к модели
config_path = "MobileNetSSD_deploy.prototxt"  # укажите путь к конфигурации

# Загрузка предварительно обученной модели MobileNetSSD для распознавания объектов
net = cv2.dnn.readNet(config_path, model_path)

# Список меток классов, которые MobileNetSSD может обнаруживать
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]

# Инициализация видеозахвата
video_capture = cv2.VideoCapture(1)

CHIP = 1
LED_OFFSET = 14
ECHO_OFFSET = 15
TRIGGER_OFFSET = 13
MIN_DIST = 150

chip = gpiod.chip(CHIP)
led = chip.get_line(LED_OFFSET)
echo = chip.get_line(ECHO_OFFSET)
trigger = chip.get_line(TRIGGER_OFFSET)

config = gpiod.line_request()
config.consumer = "Echo Pin"
config.request_type = gpiod.line_request.DIRECTION_INPUT
echo.request(config)

config = gpiod.line_request()
config.consumer = "Trigger Pin"
config.request_type = gpiod.line_request.DIRECTION_OUTPUT
trigger.request(config)

config = gpiod.line_request()
config.consumer = "LED"
config.request_type = gpiod.line_request.DIRECTION_OUTPUT
led.request(config)

def distance():
    trigger.set_value(1)
    
    time.sleep(0.00001)
    trigger.set_value(0)
    
    StartTime = time.time()
    StopTime = time.time()
    
    while echo.get_value() == 0:
        StartTime = time.time()
    
    while echo.get_value() == 1:
        StopTime = time.time()
        
    TimeElapsed = StopTime - StartTime
    distance = (TimeElapsed * 34300) / 2
    print(distance)
    time.sleep(0.1)
    
    return distance

def detect(distance, MIN_DIST):
    ret, frame = video_capture.read()

    # Получение размеров кадра
    (h, w) = frame.shape[:2]

    # Предобработка изображения для подачи в сеть
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Прямой проход по сети
    net.setInput(blob)
    detections = net.forward()

    # Обработка обнаружений
    if distance <= MIN_DIST:
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Фильтруем по классу "person"
            idx = int(detections[0, 0, i, 1])
            if confidence > 0.2 and (CLASSES[idx] == "person" or CLASSES[idx] == "boat"):  # Порог уверенности
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # Рисование прямоугольника вокруг обнаруженного человека
                label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Отображение кадра
    cv2.imshow('Video', frame)
    if distance <= MIN_DIST:
        return 1
    return 0

dist = distance()
while True:
    # Действия с нейросетью
    led.set_value(detect(dist, MIN_DIST))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    dist = distance()
    
video_capture.release()
cv2.destroyAllWindows()