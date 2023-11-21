#include <Stepper.h>

const int stepsPerRevolution = 200;

Stepper myStepper(stepsPerRevolution,8,9,10,11)

void setup(){
    myStepper.speed(60);
    serial.begin(9600);
}

void loop(){
    serial.println("clockwise");
    myStepper.step(stepsPerRevolution);
    delay(1000);
    serial.println("anticlockwise");
    myStepper.step(-stepsPerRevolution);
    delay(1000);
}