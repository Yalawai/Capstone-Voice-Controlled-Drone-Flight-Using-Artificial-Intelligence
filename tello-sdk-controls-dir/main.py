from djitellopy import Tello

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery())
print("take off", tello.takeoff())
print("land", tello.land())
