import os
from tello_sdk_controls_dir.main import SDK
from whisper_cpp.main import main
import keyboard
import threading

sdk = SDK()

def emergency_land():
    keyboard.wait("a")
    print("EMERGENCY LAND")
    sdk.DroneFlightController("land")

    os._exit(0)

threading.Thread(target=emergency_land, daemon=True).start()
while True:
    voiceCommand = main()
    sdk.DroneFlightController(voiceCommand)