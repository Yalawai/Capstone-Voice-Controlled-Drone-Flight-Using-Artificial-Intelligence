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
    def DroneSystemInformation(self):
        try:
            return self.tello.get_current_state()
        except Exception as e:
            print("Drone System Diagnostics Failed", e)
            return None


    #This take a picture and saves it
    def TakePicture(self):
        try:
            # Turns on camera
            self.tello.streamon()
            time.sleep(2)
            frame = None
            # Extracts image from video stream
            frame_read = self.tello.get_frame_read()
            for x in range(5):
                frame = frame_read.frame
                time.sleep(0.1)
            # Save image to jpg
            sucesss = cv2.imwrite("test.jpg", frame)
            print("sucesss")
        finally:
            # Turns off camera
            self.tello.streamoff()



    def DroneFlightController(self, action_dict):
        try:
            action = action_dict.get('action','UNKNOWN')
            numbers = action_dict.get('value',None)

            if action == "takeoff":
                self.tello.takeoff()
                print("takeoff")
            elif action == "land":
                self.tello.land()
                print("land")
            elif action == "up" and numbers < 30:
                self.tello.move_up(numbers)
            elif action == "down" and numbers < 30:
                self.tello.move_down(numbers)
            elif action == "left" and numbers < 30:
                self.tello.move_left(numbers)
            elif action == "right" and numbers < 30:
                self.tello.move_right(numbers)
            elif action == "rotateclockwise" and numbers < 30:
                self.tello.rotate_clockwise(numbers)
            elif action == "rotatecounterclockwise" and numbers < 30:
                self.tello.rotate_counter_clockwise(numbers)
            elif action == "motoroff" :
                self.tello.turn_motor_off()
            elif action == "UNKNOWN" :
                print("Unknown command")
            else:
                print("Error")


        except Exception as e:
            print("Drone Flight Control Failed:",e)
#sdk = SDK()
#print(sdk.DroneSystemInformation())
#sdk.DroneFlightController("takeoff")
#sdk.DroneFlightController("land")