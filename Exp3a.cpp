int trigPin=2;
int echoPin=3;

void setup(){
    Serial.begin(9600);
    pinMode(trigPin,OUTPUT);
    pinMode(echoPin,INPUT);
}

void loop(){
    displayWrite(trigPin,HIGH);
    delay(100);
    displayWrite(trigPin,LOW);
    long duration = pulseIn(echoPin,HIGH);
    float distance = duration * 0.034/2;
    Serial.println(distance);
}