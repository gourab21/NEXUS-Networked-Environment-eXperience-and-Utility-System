# NEXUS — Networked Environment eXperience and Utility System

**NEXUS** is a Windows desktop prototype for **camera-based room occupancy monitoring and zone-aware automation**. It displays a live RTSP/webcam feed, lets an operator draw seating-area polygons, counts YOLO person detections by zone, and can send zone-light brightness commands to an ESP32 over USB serial. Optional face registration and attendance logging are also included.

This is a demonstration/research prototype. It has not been benchmarked on Raspberry Pi, and its counts or face matches should not be treated as authoritative without human review.

## Features

- RTSP stream or local webcam input.
- Editable, saved polygon zones and per-zone person counts.
- YOLO11n person detection using Ultralytics, configured for CPU inference.
- Optional YuNet/SFace face registration and sampled attendance. Multiple enrollment images per person are supported.
- Optional ESP32 serial control for small LEDs, with zone-to-GPIO mapping, event schedules, occupancy response, and gradual brightness transitions.
- Local CSV attendance log.

## Requirements

- Windows 10/11, Python 3.11, and PowerShell.
- Network access to the camera stream; the phone and PC should be reachable on the same network for a Wi-Fi camera.
- Optional lighting demo: ESP32 DevKit/WROOM, USB data cable, Arduino IDE with ESP32 board support, and small LEDs with current-limiting resistors.

The app uses OpenCV, Ultralytics, NumPy, Pillow, and PySerial. The YOLO weights (`yolo11n.pt`) are fetched by Ultralytics on first launch if they are not present locally, so internet access is needed for that first download. The optional face models are stored in `models/`.

## Install and run

Open PowerShell in the cloned project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks virtual-environment activation, run the app using the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Camera and zones

1. Start an RTSP stream on the phone/camera and confirm the PC can access its URL. The app starts with the example URL `rtsp://192.168.137.231:8554/live/0`; replace it with your current stream URL if the address differs. A webcam can be selected by entering its device index, such as `0`.
2. Select **Connect**. Wait for the camera and detector status to become ready.
3. Select **New zone**, enter a zone name, and click at least three polygon vertices on the camera image. Press **Enter** or right-click to finish that polygon.
4. Repeat for each seating area and select **Save zones**. Zone geometry is saved locally in `zones.json` using camera-frame pixel coordinates. If the camera position, framing, or resolution changes, redraw the zones.

Person counts use the detected person's lower-center point to decide which polygon contains them. Occlusion, camera angle, lighting, crowding, and small or distant people can affect the result.

## Optional face enrollment and attendance

Face attendance is optional. Enroll only people who have agreed to this use. Add one or more clear face images in the app's **Register a person** control, or arrange files like this:

```text
people/
  Gourab-Das/
    Gourab-Das_001.jpg
    Gourab-Das_002.jpg
  Pavan/
    Pavan_001.jpg
```

Enable face attendance in the app to sample and match faces using the YuNet detector and SFace embeddings. Attendance sampling is configured every two seconds. `attendance.csv` records daily recognized samples and related zone information; it is local runtime data and is excluded from Git. A wide room view may not provide enough face detail for reliable identification. Review records manually and follow applicable consent and privacy requirements.

## Optional ESP32 LED demonstration

The Python app communicates with the ESP32 over **USB serial** (CP2102 or compatible). Wi-Fi is not required for this PC demonstration.

1. Install Arduino IDE and the ESP32 board package.
2. Open `esp32_lighting_controller/esp32_lighting_controller.ino`, select the correct ESP32 board and COM port, then upload the sketch. Close Arduino Serial Monitor before the app connects to that port.
3. Connect each **bare LED** through its own 220–330 Ω series resistor to a supported ESP32 GPIO and connect the LED cathode to GND. The sketch supports GPIOs 18, 19, 21, 22, 23, 25, 26, 27, 32, and 33.
4. In **Lighting & events**, select the COM port and map each zone to the GPIO wired to its LED. Save settings and connect the ESP32.

Do not connect mains bulbs, room lights, fans, or other high-voltage equipment directly to ESP32 GPIO pins. Real building equipment needs appropriately rated, isolated driver hardware and safety review. The prototype only demonstrates low-voltage LEDs.

## Project files and local data

```text
app.py                                  Tkinter camera, zones, detection, attendance, and lighting UI
requirements.txt                        Python dependencies
esp32_lighting_controller/              ESP32 serial/PWM sketch
models/                                 YuNet and SFace ONNX models for optional attendance
deliverables/                           Project paper and presentation
zones.json                              Local camera zone layout (created/saved by the app; ignored by Git)
lighting_config.json                    Local room, event, COM-port, and GPIO settings (ignored by Git)
people/                                 Local enrollment photos (ignored by Git)
attendance.csv                          Local attendance history (ignored by Git)
```

The `.gitignore` intentionally excludes local camera layouts, serial/event settings, attendance logs, and face photos. Do not commit identifiable images or attendance records. YOLO weight files and temporary build/render folders are also ignored.

## Current scope and next work

Demonstrated scope includes live camera preview, polygon-based zone occupancy, optional face enrollment/attendance, and USB serial control of small ESP32 LEDs. The larger goal is a cost-conscious smart-room system that could later coordinate lighting, fan speed, air conditioning, displays, projectors, and other equipment by zone. Those interfaces, Raspberry Pi performance, detection accuracy, energy use, and savings still need controlled testing; they are not established results of this prototype.
