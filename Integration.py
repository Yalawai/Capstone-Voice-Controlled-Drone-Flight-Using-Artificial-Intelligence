from tello_sdk_controls_dir.main import SDK
from whisper_cpp.main import main

sdk = SDK()
voiceCommand = main()

sdk.DroneFlightController(voiceCommand)
sdk.DroneFlightController("land")
