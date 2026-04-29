# Voice-Controlled Drone Flight Using Artificial Intelligence

## Overview

This project demonstrates an **AI-powered drone control system** that
enables a drone to interpret **voice commands**, analyze its
environment, and execute safe navigation actions autonomously.

The system integrates: - Voice recognition - Vision-based scene
understanding - AI planning and decision-making - Real-time drone
control via SDK

The drone continuously captures visual input, evaluates its
surroundings, and dynamically plans safe movement toward a goal while
avoiding obstacles.

## Features

-   **Voice Command Input**
    -   Users provide high-level instructions (for example, "go to the
        chair")
-   **AI-Based Planning**
    -   Generates structured navigation plans using environment
        understanding
    -   Produces actions, risk levels, and goal status
-   **Vision Processing**
    -   Captures real-time frames from the drone camera
    -   Extracts scene information for planning and obstacle detection
-   **Obstacle Avoidance**
    -   Performs safety checks before executing each movement
    -   Prevents unsafe or risky actions
-   **Closed-Loop Control**
    -   Replans after every movement using updated environment data
-   **Safety Mechanisms**
    -   Keyboard kill switch (ESC)
    -   Keepalive system to prevent auto-landing

## Project Structure

``` text
.
├── Integration.py
├── tello_sdk_controls_dir/
├── vision_action_controller_dir/
├── whisper_cpp/
├── yolo_object_detection_dir/
├── requirements.txt
├── graph.png
└── README.md
```

## Setup Instructions

### Clone the Repository

``` bash
git clone https://github.com/Yalawai/Capstone-Voice-Controlled-Drone-Flight-Using-Artificial-Intelligence.git
cd Capstone-Voice-Controlled-Drone-Flight-Using-Artificial-Intelligence
```

### Create a Virtual Environment

``` bash
python -m venv .venv
```

### Activate the Environment

**Windows**

``` bash
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

### Install Dependencies

``` bash
pip install -r requirements.txt
```

## Running the Project

``` bash
python Integration.py
```

## Safety Considerations

-   Test in an open indoor space
-   Maintain line of sight with the drone
-   Be ready to press ESC for emergency stop
-   Avoid operating near people or fragile objects


