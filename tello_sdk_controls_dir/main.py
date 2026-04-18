import numpy as np
from djitellopy import Tello
import time
import cv2
import base64


class SDK:
    def __init__(self):
        try:
            self.tello = Tello()
            self.tello.connect()
            self.tello.streamon()
            self.frame_read = self.tello.get_frame_read()
            time.sleep(2)
            print("Drone Connected")

        except:
            print("drone Not connected")

    def DroneSystemInformation(self):
        try:
            self.tello.send_command_without_return("command")
            state = self.tello.get_current_state()
            if state:
                print(
                    f"[TELEMETRY] "
                    f"Batt: {state.get('bat', '?')}% | "
                    f"Alt: {state.get('h', '?')}cm | "
                    f"Temp: {state.get('templ', '?')}-{state.get('temph', '?')}°C | "
                    f"Pitch: {state.get('pitch', '?')} Roll: {state.get('roll', '?')} Yaw: {state.get('yaw', '?')}"
                )
            return state
        except Exception as e:
            print("[TELEMETRY] failed:", e)
            return None

    def TakePicture(self):
        try:
            frame = self.frame_read.frame
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=15)
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            frame = cv2.filter2D(frame, -1, kernel)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            print("Picture Captured")
            return image_b64
        except Exception as e:
            print("Camera Failed ", e)
            return None

    def DroneFlightController(self, action, numbers):
        _MOVEMENT_ACTIONS = {
            "move_forward", "move_back", "move_left", "move_right",
            "move_up", "move_down", "rotate_clockwise", "rotate_counter_clockwise"
        }
        if action in _MOVEMENT_ACTIONS:
            if numbers < 20:
                print(f"[SDK] Skipping {action}: value {numbers} is below minimum 20")
                return
            if numbers > 500:
                numbers = 500
                print(f"[SDK] Clamped {action} value to 500")

        try:
            if action == "takeoff":
                self.tello.takeoff()
                print("takeoff")
            elif action == "land":
                self.tello.land()
                print("land")
            elif action == "move_up":
                self.tello.move_up(numbers)
                print("move_up", numbers)
            elif action == "move_down":
                self.tello.move_down(numbers)
                print("move_down", numbers)
            elif action == "move_forward":
                self.tello.move_forward(numbers)
                print("move_forward", numbers)
            elif action == "move_back":
                self.tello.move_back(numbers)
                print("move_back", numbers)
            elif action == "move_left":
                self.tello.move_left(numbers)
                print("move_left", numbers)
            elif action == "move_right":
                self.tello.move_right(numbers)
                print("move_right", numbers)
            elif action == "rotate_clockwise":
                self.tello.rotate_clockwise(numbers)
                print("rotate_clockwise", numbers)
            elif action == "rotate_counter_clockwise":
                self.tello.rotate_counter_clockwise(numbers)
                print("rotate_counter_clockwise", numbers)
            elif action == "hover":
                print("hover")
            else:
                print(f"[SDK] Unknown action: {action}")
        except Exception as e:
            print("Drone Flight Controller Failed", e)

    def ShutDown(self):
        try:
            self.tello.streamoff()
            self.tello.end()
        except Exception as e:
            print("Shutdown Failed", e)