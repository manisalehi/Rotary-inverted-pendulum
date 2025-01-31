from controller import Robot
import random
import numpy.random as rand
import math 

import numpy as np
import pandas as pd

TIME_STEP = 8          #Time step for phhyscial simulation
SAMPLE_TIME = TIME_STEP #Time step for Sensor/controller 

class PID():
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.dt = dt

        self.previous_error = 0
        self.integral = 0
 
     # #Behaves like the TF given thhe error will calculate the control signal
    def compute(self, dt, error):
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I_out = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        D_out = self.Kd * derivative
        
        # Compute total output
        output = P_out + I_out + D_out
        
        # Update previous error
        self.previous_error = error
        
 
        return output
        
    def saturate(self, signal, limits):
       if abs(signal) > abs(limits): 
           return abs(limits) * math.copysign(1,signal)
       
       return signal 
  
def Noise(signal, sigma=0.05):
    return rand.normal(signal/10 ,sigma,1)[0]

 
robot = Robot()

#Motor
m = robot.getDevice("rotational motor")
m.setPosition(float('inf'))                   #Must be set or the setVelocity wont work

#Sensor
sensor = robot.getDevice("position sensor")
sensor.enable(SAMPLE_TIME)

#Seting up the PID
controller = PID(Kp= 7.757507,Ki=48.22523,Kd=0.205, dt=SAMPLE_TIME)
# controller = PID(Kp= 10,Ki=50,Kd=5, dt=SAMPLE_TIME)
# controller = PID(Kp= -15.7179,Ki=0,Kd=-12.1, dt=SAMPLE_TIME)
# controller = PID(Kp= 10,Ki=50,Kd=5, dt=SAMPLE_TIME)

print(controller.Kp, controller.Kd )

i=0
while (robot.step(TIME_STEP) != -1):
    #Itteration counter
    i +=1
    
    #Finding the e(t) = 0 - y(t)
    error =-1 * sensor.getValue()
    
    
    #Finding the controller's signal part
    u = controller.compute(error=error, dt=SAMPLE_TIME)
    #Saturate the signal
    u_sat = controller.saturate(signal=u, limits=90) 
    
    #Noise
    # d = Noise(signal = u_sat)
    d= 0
   
    
    #Applying noise every 10 steps
    if i%10 != 0:
        d = 0
    
    #Determining the signals 
    #print("Saturated control signal:", u_sat)
    #print("noise:", d)
    
    #Theta1_dot
    m.setVelocity(u_sat + d)
    
    #Finding the simulation time
    #print(robot.getTime())
  