// USB serial PWM controller for the Hall Occupancy Python app.
// Protocol: PING -> PONG; SET <GPIO> <brightness 0..100> updates PWM silently.
// Bare LEDs must use current-limiting resistors and share ESP32 GND.

const uint8_t ALLOWED_PINS[] = {18, 19, 21, 22, 23, 25, 26, 27, 32, 33};
const size_t PIN_COUNT = sizeof(ALLOWED_PINS) / sizeof(ALLOWED_PINS[0]);
bool pinInitialized[PIN_COUNT] = {false};
String commandBuffer;

int allowedPinIndex(int pin) {
  for (size_t i = 0; i < PIN_COUNT; ++i) {
    if (ALLOWED_PINS[i] == pin) return static_cast<int>(i);
  }
  return -1;
}

void handleCommand(String command) {
  command.trim();
  command.toUpperCase();
  if (command == "PING") {
    Serial.println("PONG");
    return;
  }

  int pin = -1;
  int brightness = -1;
  if (sscanf(command.c_str(), "SET %d %d", &pin, &brightness) != 2) {
    Serial.println("ERR use: SET <GPIO> <0..100> or PING");
    return;
  }
  const int index = allowedPinIndex(pin);
  if (index < 0 || brightness < 0 || brightness > 100) {
    Serial.println("ERR invalid GPIO or brightness");
    return;
  }
  if (!pinInitialized[index]) {
    pinMode(pin, OUTPUT);
    pinInitialized[index] = true;
  }
  analogWrite(pin, map(brightness, 0, 100, 0, 255));
}

void setup() {
  Serial.begin(115200);
  commandBuffer.reserve(48);
  for (uint8_t pin : ALLOWED_PINS) {
    pinMode(pin, OUTPUT);
    analogWrite(pin, 0);
  }
  Serial.println("ESP32_LIGHTING_READY");
}

void loop() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      if (commandBuffer.length() > 0) {
        handleCommand(commandBuffer);
        commandBuffer = "";
      }
    } else if (commandBuffer.length() < 47) {
      commandBuffer += c;
    }
  }
}
