int ledpin;
void pattern1();
void pattern2();

void setup(){
    serial.begin(9600);
    for(ledpin = 2;ledpin<6;ledpin++){
        pinMode(ledpin,OUTPUT);
    }
    serial.println("Choose pattern 1 or 2");
}

void pattern1(){
    for(ledpin=2;ledpin<6;ledpin++){
        displayWrite(ledpin,HIGH);
    }
    exit();
}

void pattern2(){
    for(ledpin=2;ledpin<6;ledpin++){
        displayWrite(ledpin,HIGH);
        delay(500);
        displayWrite(ledpin,LOW);
    }
    exit();
}

void loop(){
    int inp = serial.parseInt();
    if(inp==1){
        pattern1();
    }
    else if(inp==2){
        pattern2();
    }
    else {
        serial.println("Input is invalid");
    }
}