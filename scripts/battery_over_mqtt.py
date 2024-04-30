import os
import sys

# Add the root directory of your project to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time

import paho.mqtt.client as mqtt
import psutil

# MQTT Broker details
broker_address = "homeassistant.local"
broker_port = 1883
topic = "x230/battery_state"

def get_battery_state():
    battery = psutil.sensors_battery()
    return battery.percent

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print("Failed to connect, return code %d\n", rc)

def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT Broker")

def on_publish(client, userdata, mid):
    print("Message Published")

client = mqtt.Client("BatteryPublisher")

client.username_pw_set("virtual_user", "01steffen01")

client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_publish = on_publish

client.connect(broker_address, broker_port, 60)
client.loop_start()  # Start the loop

try:
    while True:
        battery_state = get_battery_state()
        info = client.publish(topic, battery_state)
        info.wait_for_publish()
        time.sleep(60)
except KeyboardInterrupt:
    client.loop_stop()  # Stop the loop
    client.disconnect()
