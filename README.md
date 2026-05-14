# Voice-Controlled Drone Flight Using Artificial Intelligence

Working Demo: Goal search for a white table and land on the table 

https://github.com/user-attachments/assets/87101d75-5211-4971-aed9-81c5ea1d2924

Some additional test run can be found using this link:

[Additional Test Run](https://unicanberraedu-my.sharepoint.com/:f:/g/personal/u3231348_uni_canberra_edu_au/IgBME5XqGN79S45BdGrLH6VxAXTENCY-nrf03g2vAza3G50?e=pYGiID)


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
    -   Prevents unsafe or risky actions
-   **Closed-Loop Control**
    -   Replans after every movement using updated environment data
-   **Safety Mechanisms**
    -   Keyboard kill switch (ESC)
    -   Keepalive system to prevent auto-landing
 
## System Architecture

<img width="2490" height="1453" alt="image" src="https://github.com/user-attachments/assets/095dff09-ee2d-4440-998e-523a0c9ba1d3" />


## Prerequisites & Setup

### Hardware Requirements

-    Drone: Ryze Tello (ensure it is fully charged).
-    Connection: A dedicated Wi-Fi card to connect to the Tello access point.
-    Audio: A high-quality microphone for voice commands.

### Software Requirements

-    Python 3.10+

-    Google API Key: Required for Gemini/Vision processing.

-    C++ Build Environment: (CMake + Compiler) for whisper.cpp.


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

## API Key Configuration

Create a .env file or export your Google API key

Get a google api key from: [https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key)

# Voice Model Setup (Whisper.cpp)

This project uses **whisper.cpp** for speech-to-text conversion. By default, a lightweight model is used for real-time performance. However, you can improve accuracy by manually downloading and using a higher-quality model.

## Improve Voice Recognition Accuracy

If you want better transcription accuracy, you can switch to a larger Whisper model (e.g., `base`, `small`, or `medium`).

Higher models improve accuracy but may reduce speed and increase memory usage.

---

## Manual Model Download

You must manually download the Whisper model in **ggml format** and place it inside the appropriate directory used by the project.

Change directory to whisper_cpp:

```bash
cd whisper_cpp
```

### Step 1: Download a Model

You can download models from the official repository:

[https://huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main)

For example, to download the **base English model**:

```bash
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

Or using the provided script:

```bash
./models/download-ggml-model.sh base.en
```
Or Manually Download the model using the above link:

---

### Step 2: Place the Model in the Project

After downloading, place the `.bin` file into the expected model directory used by this project (typically inside the `whisper_cpp/` or models folder).

Example:

```
whisper_cpp/models/ggml-base.en.bin
```

---

### Step 3: Update Configuration 

Update the model path in your Whisper integration code inside:

```
whisper_cpp/main.py
```

Change model path to the downloaded model name:

<img width="559" height="79" alt="image" src="https://github.com/user-attachments/assets/7b435c68-372b-4e9a-9038-57ba88a82c30" />

Or

Look for where the model is loaded and replace the path with your chosen model.

#### **you need a C++ build environment + CMake + a compiler toolchain set up correctly**

Now build the whisper-cli:
```
# build the project
cmake -B build
cmake --build build -j --config Release
```

---

## Recommended Models

| Model | Speed | Accuracy |
|------|------|---------|
| tiny | Very fast | Low |
| base | Fast | Good balance |
| small | Medium | Better accuracy |
| medium | Slow | High accuracy |

---

## Notes

- Larger models improve recognition in noisy environments.
- For real-time drone control, `base.en` or `small.en` is usually the best trade-off.
- Ensure the model file path matches exactly in your code, otherwise loading will fail.

---

## Reference

[https://github.com/ggml-org/whisper.cpp?tab=readme-ov-file](https://github.com/ggml-org/whisper.cpp?tab=readme-ov-file)





## Operation Guide

### Pre-Flight Checklist
*   **Calibration:** If the drone has recently crashed or drifted, calibrate the IMU via the official Tello app.
*   **Environment:** Ensure the area is well-lit (the AI relies on visual telemetry) and quiet to minimize STT latency.
*   **Battery:** Do not attempt flight below 20% battery.



## Safety Considerations

-   Test in an open indoor space
-   Maintain line of sight with the drone
-   Be ready to press ESC for emergency stop
-   Avoid operating near people or fragile objects

## Limitations & Maintenance
### Known Issues

-    Hardware Drift: Due to the Tello's lack of GPS, internal IMU drift may occur during long sessions.

-    Latency: Processing voice and vision via API calls introduces a slight delay between command and action.

-    Lighting: In low-light conditions, scene understanding accuracy drops significantly.

## Running the Project

Run the following command from the root folder
``` bash
python -m Integration
```


