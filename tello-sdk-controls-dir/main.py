from djitellopy import Tello

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery())
