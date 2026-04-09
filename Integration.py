import os
from tello_sdk_controls_dir.main import SDK
from whisper_cpp.main import main
import threading
from vision_action_controller_dir.main import process_drone_cycle


sdk = SDK()


threading.Thread(target=sdk.emergency_land(), daemon=True).start()
while True:
    voiceCommand = main()

    proposal = process_drone_cycle(voiceCommand, None, None)
    amount = 0
    print("AI proposed:", proposal)
    if proposal.get("confidence", 0) >= 0.7:

        sdk.DroneFlightController(proposal["action"], proposal["value"])