#include <WiFi.h>
#include <esp_now.h>

#define TRIG_PIN 12
#define ECHO_PIN 4

// 110211'60 % 50 + 5 = 15
#define SLEEP_TIME 15

// MAC addr
uint8_t broadcastAddress[] = {0x8C, 0xAA, 0xB5, 0x84, 0xFB, 0x90}; 

// radio time
unsigned long radioStart;
unsigned long radioEnd;

esp_now_peer_info_t peerInfo;

// Receiving Callback
void OnDataRecv(const uint8_t * mac, const uint8_t *data, int len) { // used in vscode
//void OnDataRecv(const esp_now_recv_info * mac, const uint8_t *data, int len) {
  Serial.print("< Message received: ");
  char receivedString[len];
  memcpy(receivedString, data, len);
  Serial.println(String(receivedString));
}

// Sending callback
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // end of trans
  radioEnd = micros(); //us
  Serial.print("trans time: ");
  Serial.print(radioEnd - radioStart);
  Serial.println(" us");
}

void setup() {
  // at this time esp32 start and sensor start
  unsigned long startTime = millis();
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  Serial.begin(115200);

  // send signal
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // read ECHO high-time (us)
  long duration = pulseIn(ECHO_PIN, HIGH);
  
  // get distance (cm)
  float distance = duration * 0.034 / 2;

  Serial.print("> Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // check distance
  String message;
  if( distance > 50 ) message = "FREE";
  else message = "OCCUPIED";

  // at this time WiFi start
  unsigned long startWifi = millis();
  Serial.print("busy time: ");
  Serial.print(startWifi - startTime);
  Serial.println(" ms");

  // initialize Wi-Fi
  WiFi.mode(WIFI_STA);
  esp_now_init();
  esp_now_register_send_cb(OnDataSent);
  esp_now_register_recv_cb(OnDataRecv);

  // peer Registration
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);

  // start trans message
  WiFi.setTxPower(WIFI_POWER_2dBm);
  radioStart = micros(); //us
  Serial.print("wifi time: ");
  Serial.print(radioStart/1000 - startWifi);
  Serial.println(" ms");
  esp_now_send(broadcastAddress, (uint8_t*)message.c_str(),message.length() + 1);
  Serial.println("> " + message);

  // sleep
  esp_sleep_enable_timer_wakeup(SLEEP_TIME * 1000000); // us to s
  Serial.println("---- will sleep " + String(SLEEP_TIME) + "s ----");
  esp_deep_sleep_start();

}

void loop() {
}
