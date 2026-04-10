from djitellopy import Tello
import time
import cv2
import re
import base64
import os
import msvcrt
import threading

from sentry_sdk.utils import single_exception_from_error_tuple


class SDK:
    def __init__(self):
        self._lock = threading.Lock()
        try:
            self.tello = Tello()
            self.tello.connect()
            self.tello.streamon()
            self.frame_read = self.tello.get_frame_read()
            time.sleep(2)
            print("Drone Connected")

        except:
            print("drone Not connected")

    def emergency_land(self):
        print("Press 'a' to EMERGENCY LAND")
        while True:
            if msvcrt.kbhit() and msvcrt.getwch() == 'a':
                break
            time.sleep(0.05)
        print("EMERGENCY LAND")
        self.DroneFlightController("land", 0)
        os._exit(0)

    def emergency_kill(self):
        try:
            print("Emergency Kill")
            self.tello.emergency()
        except Exception as e:
            print("Emergency Kill Failed", e)

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
            frame = self.frame_read.frame
            # Convert BGR → RGB (optional but better for vision models)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Encode once here
            _, buffer = cv2.imencode('.jpg', frame_rgb)
            print("Turn into jpg")
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            print("decoded to b64")
            print("Picture Captured")
            return image_b64
        except Exception as e:
            print("Camera Failed ", e)
            return None



    def telemetry_thread(self, stop_event):
        while not stop_event.wait(5):
            try:
                with self._lock:
                    self.tello.send_command_without_return("command")
                state = self.DroneSystemInformation()
                if state:
                    print(
                        f"[TELEMETRY] "
                        f"Batt: {state.get('bat', '?')}% | "
                        f"Alt: {state.get('h', '?')}cm | "
                        f"Temp: {state.get('templ', '?')}-{state.get('temph', '?')}°C | "
                        f"Pitch: {state.get('pitch', '?')} Roll: {state.get('roll', '?')} Yaw: {state.get('yaw', '?')}"
                    )
            except Exception as e:
                print("[TELEMETRY] failed:", e)

    def DroneFlightController(self, action, numbers):
        with self._lock:
            try:
                if action == "takeoff":
                    self.tello.takeoff()
                    print("takeoff")
                elif action == "land":
                    self.tello.land()
                    print("land")
                elif action == "move_up":
                    self.tello.move_up(numbers)
                    print("move_up")
                elif action == "move_down":
                    self.tello.move_down(numbers)
                    print("move_down")
                elif action == "move_forward":
                    self.tello.move_forward(numbers)
                    print("move_forward")
                elif action == "move_backward":
                    self.tello.move_back(numbers)
                    print("move_backward")
                elif action == "move_left":
                    self.tello.move_left(numbers)
                    print("move_left")
                elif action == "move_right":
                    self.tello.move_right(numbers)
                    print("move_right")
                elif action == "rotate_clockwise":
                    self.tello.rotate_clockwise(numbers)
                    print("rotate_clockwise")
                elif action == "rotate_counter_clockwise":
                    self.tello.rotate_counter_clockwise(numbers)
                    print("rotate_counter_clockwise")
                elif action == "hover":
                    print("hover")
                else:
                    print(action)
            except Exception as e:
                print("Drone Flight Controller Failed", e)

    def ShutDown(self):
        try:
            self.tello.streamoff()
            self.tello.end()
        except Exception as e:
            print("Shutdown Failed", e)

