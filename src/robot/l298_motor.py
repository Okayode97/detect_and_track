
from typing import Optional
from dataclasses import dataclass
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


Reading up on 4WD Robots with Omni wheels
- Consideration on how to make sharper turns
- possible need to set to individual motors to run at different speeds
- Consideration on control system which determine write function call to make each time.

- possible solution
    - single control system which an input direction and speed to travel in
        - input direction is broken down into components which then determine how to control each individual motors 
"""

GPIO.setmode(GPIO.BCM)

class Motor:
    def __init__(self, dir_pin_1: int, dir_pin_2: int, motor_pwm_pin: Optional[int] = None):
        self.dir_pin_1 = dir_pin_1
        self.dir_pin_2 = dir_pin_2
        self.motor_pwm_pin = None
        self.motor_pwm = None

        GPIO.setup(self.dir_pin_1, GPIO.OUT)
        GPIO.setup(self.dir_pin_2, GPIO.OUT)

        if motor_pwm_pin:
            self.motor_pwm_pin = motor_pwm_pin
            GPIO.setup(self.motor_pwm_pin, GPIO.OUT)
            self.motor_pwm = GPIO.PWM(self.motor_pwm_pin, 100)
            self.motor_pwm.start(50)


    def set_direction(self, forward: bool):
        GPIO.output(self.dir_pin_1, GPIO.HIGH if forward else GPIO.LOW)
        GPIO.output(self.dir_pin_2, GPIO.LOW if forward else GPIO.HIGH)

    def set_duty_cycle(self, duty_cycle: float):
        self.motor_pwm.ChangeDutyCycle(duty_cycle)


@dataclass
class MotorConfig:
    fl_motor_dir_pin_1: int
    fl_motor_dir_pin_2: int
    fl_motor_pwm_pin: Optional[int]

    fr_motor_dir_pin_1: int
    fr_motor_dir_pin_2: int
    fr_motor_pwm_pin: Optional[int]

    rl_motor_dir_pin_1: int
    rl_motor_dir_pin_2: int
    rl_motor_pwm_pin: Optional[int]

    rr_motor_dir_pin_1: int
    rr_motor_dir_pin_2: int
    rr_motor_pwm_pin: Optional[int]


class Robot:

    def __init__(self, motor_config: MotorConfig):
        self.fl_motor = Motor(motor_config.fl_motor_dir_pin_1, motor_config.fl_motor_dir_pin_2, motor_config.fl_motor_pwm_pin)
        self.fr_motor = Motor(motor_config.fr_motor_dir_pin_1, motor_config.fr_motor_dir_pin_2, motor_config.fr_motor_pwm_pin)
        self.rl_motor = Motor(motor_config.rl_motor_dir_pin_1, motor_config.rl_motor_dir_pin_2, motor_config.rl_motor_pwm_pin)
        self.rr_motor = Motor(motor_config.rr_motor_dir_pin_1, motor_config.rr_motor_dir_pin_2, motor_config.rr_motor_pwm_pin)
    

    def set_fl_speed_and_direction(self, direction: bool, speed: float):
        self.fl_motor.set_direction(direction)
        self.fl_motor.set_duty_cycle(speed)

    def set_fr_speed_and_direction(self, direction: bool, speed: float):
        self.fr_motor.set_direction(direction)
        self.fr_motor.set_duty_cycle(speed)
    
    def set_rl_speed_and_direction(self, direction: bool, speed: float):
        self.rl_motor.set_direction(direction)
        self.rl_motor.set_duty_cycle(speed)
    
    def set_rr_speed_and_direction(self, direction: bool, speed: float):
        self.rr_motor.set_direction(direction)
        self.rr_motor.set_duty_cycle(speed)
    
    def move_forward(self, direction: bool, speed: float):
        self.set_fl_speed_and_direction(direction, speed)
        self.set_fr_speed_and_direction(direction, speed)
        self.set_rl_speed_and_direction(direction, speed)
        self.set_rr_speed_and_direction(direction, speed)
    
    def move_sideways(self, direction: bool, speed: float):
        self.set_fl_speed_and_direction(direction, speed)
        self.set_fr_speed_and_direction(not direction, speed)
        self.set_rl_speed_and_direction(not direction, speed)
        self.set_rr_speed_and_direction(direction, speed)

    def move_diagonal_fr(self, direction: bool, speed: float):
        self.set_fl_speed_and_direction(direction, speed)
        self.set_fr_speed_and_direction(direction, 0)
        self.set_rl_speed_and_direction(direction, 0)
        self.set_rr_speed_and_direction(direction, speed)
    
    def move_diagonal_fl(self, direction: bool, speed: float):
        self.set_fl_speed_and_direction(direction, 0)
        self.set_fr_speed_and_direction(direction, speed)
        self.set_rl_speed_and_direction(direction, speed)
        self.set_rr_speed_and_direction(direction, 0)

    def turn_around(self, direction: bool, speed: float):
        self.set_fl_speed_and_direction(direction, speed)
        self.set_fr_speed_and_direction(not direction, speed)
        self.set_rl_speed_and_direction(direction, speed)
        self.set_rr_speed_and_direction(not direction, speed)

    def turn_rear_axis(self, direction: bool, speed: float):
        self.set_fl_speed_and_direction(direction, speed)
        self.set_fr_speed_and_direction(not direction, speed)
        self.set_rl_speed_and_direction(direction, 0)
        self.set_rr_speed_and_direction(direction, 0)

    def concerning(self, direction: bool, speed: float):
        if direction:
            self.set_fl_speed_and_direction(direction, speed)
            self.set_fr_speed_and_direction(direction, 0)
            self.set_rl_speed_and_direction(direction, speed)
            self.set_rr_speed_and_direction(direction, 0)
        else:
            self.set_fl_speed_and_direction(direction, 0)
            self.set_fr_speed_and_direction(direction, speed)
            self.set_rl_speed_and_direction(direction, 0)
            self.set_rr_speed_and_direction(direction, speed)
"""
test
FR
- 26, 19, 13 (pwm pin)

RR
- 16 (pwn in), 20, 21

FL
- 4(PWM pin), 2, 3

RL
- 14(PWM pin), 15, 18
"""

test_config = MotorConfig(fl_motor_dir_pin_1=23,
                          fl_motor_dir_pin_2=24,
                          fl_motor_pwm_pin=18,

                          rr_motor_dir_pin_1=6,
                          rr_motor_dir_pin_2=5,
                          rr_motor_pwm_pin=13,

                          rl_motor_dir_pin_1=8,
                          rl_motor_dir_pin_2=25,
                          rl_motor_pwm_pin=12,

                          fr_motor_dir_pin_1=16,
                          fr_motor_dir_pin_2=20,
                          fr_motor_pwm_pin=19
                          )

test = Robot(motor_config=test_config)

while True:
    test.turn_rear_axis(True, 25)

GPIO.cleanup()