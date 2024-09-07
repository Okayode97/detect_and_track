
from typing import Optional
import RPi.GPIO as GPIO       

"""
Reminding myself on L298N
- https://lastminuteengineers.com/l298n-dc-stepper-driver-arduino-tutorial/
- https://www.electronicshub.org/raspberry-pi-l298n-interface-tutorial-control-dc-motor-l298n-raspberry-pi/

- Motor driver can control the speed and direction of two DC motors
- direction controlled by setting input pins for (1 & 2) for motor A and (3 & 4) for motor B
    - high-low combination of input pins determine motor direction
- speed controlled via PWM by EN pins for motor A & B.
    - PWM, averages input voltage by sending series of ON-OFF pules.
    - Average voltage is proportional to width of pules. Higher duty cycles --> higher average voltage.


Technical specification
L298N Motor driver has a supply voltage of 5V to 35V and is capable of 2A continious current per channel.

- VS powers internal H-Bridge which drives motor and accept input voltage from 5 - 12V
- VSS powers logic circuitry within the L298N IC, can range between 5V - 7V..

- PWM pins are by default pulled HIGH, resulting in motors moving receiving full input VS voltage by default.

- 78M05 5V regulator, when enabled VSS receives voltage from VS and can supply 5V at 0.5A. When disabled
5V needs to be supplied to VSS pin.
- Suggested to keep regulator in place if VS supply is less than 12V, if over 12V supplied suggested to provide
VS power seperately. 
"""


class L298N:
    def __init__(self, dir_pin_1: int, dir_pin_2: int, dir_pin_3: int, dir_pin_4: int,
                 motor_1_pwm: Optional[int] = None, motor_2_pwm: Optional[int] = None):
        # input control for each motors to control direction
        self.dir_pin_1 = dir_pin_1
        self.dir_pin_2 = dir_pin_2
        self.dir_pin_3 = dir_pin_3
        self.dir_pin_4 = dir_pin_4

        # set GPIO pins
        GPIO.setup(self.dir_pin_1, GPIO.OUT)
        GPIO.setup(self.dir_pin_2, GPIO.OUT)
        GPIO.setup(self.dir_pin_3, GPIO.OUT)
        GPIO.setup(self.dir_pin_4, GPIO.OUT)
        

        # set pwm pin if not None
        self.motor_1_pwm = None
        if motor_1_pwm:
            self.motor_1_pwm = motor_1_pwm
            GPIO.setup(self.motor_1_pwm, GPIO.OUT)
        

        self.motor_2_pwm = None
        if motor_2_pwm:
            self.motor_2_pwm = motor_2_pwm  
            GPIO.setup(self.motor_2_pwm, GPIO.OUT)
              
    
