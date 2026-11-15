// MIMIC arm firmware: 4 joint servos + gripper servo, commanded over USB serial.
//   "J,base,shoulder,elbow,wrist\n"  -> smooth move to those servo angles (0-180), replies "OK"
//   "G,1\n" close gripper / "G,0\n" open gripper, replies "OK"
#include <Servo.h>

const int PINS[5] = {3, 5, 6, 9, 10};      // base, shoulder, elbow, wrist, gripper
const int GRIP_OPEN = 40, GRIP_CLOSED = 100; // tune for your gripper
Servo servos[5];
int cur[5] = {90, 90, 90, 90, GRIP_OPEN};

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 5; i++) { servos[i].attach(PINS[i]); servos[i].write(cur[i]); }
  Serial.println("READY");
}

void moveTo(const int t[4]) {
  bool moving = true;
  while (moving) {
    moving = false;
    for (int i = 0; i < 4; i++) {
      if (cur[i] < t[i]) { cur[i]++; moving = true; }
      else if (cur[i] > t[i]) { cur[i]--; moving = true; }
      servos[i].write(cur[i]);
    }
    delay(12);  // lower = faster
  }
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.startsWith("J,")) {
    int a, b, c, d;
    if (sscanf(line.c_str(), "J,%d,%d,%d,%d", &a, &b, &c, &d) == 4) {
      int t[4] = {constrain(a, 0, 180), constrain(b, 0, 180), constrain(c, 0, 180), constrain(d, 0, 180)};
      moveTo(t);
      Serial.println("OK");
    } else Serial.println("ERR");
  } else if (line.startsWith("G,")) {
    cur[4] = line.substring(2).toInt() ? GRIP_CLOSED : GRIP_OPEN;
    servos[4].write(cur[4]);
    delay(400);
    Serial.println("OK");
  } else Serial.println("ERR");
}
