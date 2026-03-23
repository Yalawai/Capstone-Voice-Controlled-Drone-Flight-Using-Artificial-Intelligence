from djitellopy import Tello
import time
import cv2
import re

class SDK:
    try:
        tello = Tello()
        tello.connect()
    except:
        print("drone Not connected")

    # Drone System Diagnostics
    def DroneSystemInformation(self, infoType):
        infoType = infoType.strip().lower()
        try:
            if infoType == "battery":
                battery = self.tello.get_battery()
                return battery
            elif infoType == "temperature":
                temperature = self.tello.get_temperature()
                return temperature
            elif infoType ==  "flghttime":
                flightTime = self.tello.get_flight_time()
                return flightTime
            elif infoType == "height":
                height = self.tello.get_distance_tof()
                return height
        except Exception as e:
            print("Drone System Diagnostics Failed", e)



    #This take a picture and saves it
    def TakePicture(self):
        try:
            # Turns on camera
            self.tello.streamon()
            time.sleep(2)
            frame = None
            # Extracts image from video stream
            frame_read = self.tello.get_frame_read()
            for x in range(2):
                frame = frame_read.frame
                time.sleep(0.1)

            # Save image to jpg
            cv2.imwrite("test.jpg", frame)
            print("Photo saved to jpg")
        finally:
            # Turns off camera
            self.tello.streamoff()
            self.tello.end()


    def DroneFlightController(self, command):
        try:
            command = re.sub(r'[^a-zA-Z0-9]', '', command).lower()
            noNumberCommand = re.sub(r'[^a-zA-Z]', '', command).lower()
            digits = "".join(c for c in command if c.isdigit())
            if digits:
                numbers = int(digits)
            else:
                numbers = 0

            if noNumberCommand == "takeoff":
                self.tello.takeoff()
            elif noNumberCommand == "land":
                self.tello.land()
            elif noNumberCommand == "up" and numbers < 30:
                self.tello.move_up(numbers)
            elif noNumberCommand == "down" and numbers < 30:
                self.tello.move_down(numbers)
            elif noNumberCommand == "left" and numbers < 30:
                self.tello.move_left(numbers)
            elif noNumberCommand == "right" and numbers < 30:
                self.tello.move_right(numbers)
            elif noNumberCommand == "rotateclockwise" and numbers < 30:
                self.tello.rotate_clockwise(numbers)
            elif noNumberCommand == "rotatecounterclockwise" and numbers < 30:
                self.tello.rotate_counter_clockwise(numbers)
            elif noNumberCommand == "motoroff" :
                self.tello.turn_motor_off()
            else:
                print("Unknown command")

        except Exception as e:
            print("Drone Flight Control Failed:",e)
