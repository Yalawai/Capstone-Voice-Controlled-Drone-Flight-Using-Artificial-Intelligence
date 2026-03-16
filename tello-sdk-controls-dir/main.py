from djitellopy import Tello
import time
import cv2

tello = Tello()
tello.connect()

try:
    # Drone System Diagnostics
    print("Drone System Diagnostics")
    print("Battery:",tello.get_battery())
    print("Temperature:",tello.get_temperature())
    print("flight Time:",tello.get_flight_time())

    # Turns on camera
    tello.streamon()
    time.sleep(2)

    # Extracts image from video stream
    frame_read = tello.get_frame_read()
    frame = frame_read.frame

    # Save image to jpg
    cv2.imwrite("test.jpg", frame)
    print("Photo saved to jpg")

finally:
    # Turns off camera
    tello.streamoff()
    tello.end()