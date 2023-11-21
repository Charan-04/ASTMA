int trigPin=2;
int echoPin=3;
int buzz=4;
void setup(){
    Serial.begin(9600);
    pinMode(trigPin,OUTPUT);
    pinMode(echoPin,INPUT);
    pinMode(buzz,OUTPUT);
}

void loop(){
    displayWrite(trigPin,HIGH);
    delay(100);
    displayWrite(trigPin,LOW);
    long duration = pulseIn(echoPin,HIGH);
    float distance = duration * 0.034/2;
    if(distance<50){
        displayWrite(buzz,HIGH);
    }
    else{
        displayWrite(buzz,LOW);
    }
    Serial.println(distance);
}