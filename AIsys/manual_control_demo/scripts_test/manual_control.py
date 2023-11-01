import pygame
from lib.vehicle import Vehicle
from swarmae.SwarmAEClient import SwarmAEClient

class KeyboardController:
    def __init__(self, Vehicle):
        self.prev_accel_ = 0
        self.prev_brake_ = 0
        self.prev_steer_ = 0
        
        self.accel_ = 0
        self.brake_ = 0
        self.steer_ = 0
        
        self.hand_brake_ = 0
        
        self.Vehicle_ = Vehicle
        
        print("ManualController Ready.")
        
    def read_keyboard(self, keys):
        # W 和 S 分别表示加速和减速
        if keys[pygame.K_UP]:
            if self.prev_accel_ < 0:
                self.prev_accel_ = 0
            
            accel = self.prev_accel_ + 0.1
            accel = min(accel, 1.0) 
            self.accel_ = accel
            self.prev_accel_ = self.accel_
        elif keys[pygame.K_DOWN]:
            if self.prev_accel_ > 0:
                self.prev_accel_ = 0
                
            accel = self.prev_accel_ - 0.1
            accel = max(accel, -1.0)
            self.accel_ = accel
            self.prev_accel_ = self.accel_
        else:
            self.prev_accel_ = 0
            self.accel_ = 0

        steer_increment = 0.1
        
        # A 和 D 分别表示向左和向右转向
        if keys[pygame.K_LEFT]:
            if self.prev_steer_ > 0:
                self.prev_steer_ = 0
            else:
                self.prev_steer_ -= steer_increment
        elif keys[pygame.K_RIGHT]:
            if self.prev_steer_ < 0:
                self.prev_steer_ = 0
            else:
                self.prev_steer_ += steer_increment
        else:
            self.prev_steer_ = 0.0
        
        self.steer_ = min(1.0, max(-1.0, self.prev_steer_))  
        
        # 空格键表示拉手刹
        self.hand_brake_ = keys[pygame.K_BACKSPACE]
        
        # ESC 键表示退出 ManualControl
        break_flag = bool(keys[pygame.K_ESCAPE])
        
        # 输出控制量
        self.Vehicle_.longitudinal_control(self.accel_, self.hand_brake_)
        self.Vehicle_.lateral_control(self.steer_)
        
        print("\rControl Input: steer %.2f, accel %.2f, hand_brake %d" % (self.steer_, self.accel_, self.hand_brake_), end="")
        
        return break_flag

########################################################################################
# Main Loop
########################################################################################
AeClient = SwarmAEClient(ue_ip="localhost", ue_port=2000)

Vehicle_1 = Vehicle(Client=AeClient, vehicle_name="UGV_1", vehicle_no=1)
Vehicle_2 = Vehicle(Client=AeClient, vehicle_name="UGV_2", vehicle_no=2)
Vehicle_3 = Vehicle(Client=AeClient, vehicle_name="UGV_3", vehicle_no=3)

print("Vehilce Info: ")
print("id    : ", Vehicle_1.id_)
print("name  : ", Vehicle_1.name_)
print("N.O.  : ", Vehicle_1.no_)
print("type  : ", Vehicle_1.type_)
print("model : ", Vehicle_1.model_)
print("team  : ", Vehicle_1.team_)
print("color : ", Vehicle_1.color_)
print("width : ", Vehicle_1.width_)
print("length: ", Vehicle_1.length_)
print("height: ", Vehicle_1.height_)

ManualControl_1 = KeyboardController(Vehicle_1)
ManualControl_2 = KeyboardController(Vehicle_2)
ManualControl_3 = KeyboardController(Vehicle_3)

ManualControl   = ManualControl_1
vehicle_no = "1"

pygame.init()

screen_width  = 200
screen_height = 200
screen        = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Temp Image")

run = True
while run:
    pygame.time.delay(10)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_1]:
        ManualControl = ManualControl_1
        vehicle_no = 1
    elif keys[pygame.K_2]:
        ManualControl = ManualControl_2
        vehicle_no = 2
    elif keys[pygame.K_3]:
        ManualControl = ManualControl_3
        vehicle_no = 3
    
    break_flag = ManualControl.read_keyboard(keys)
    print(" vehicle N.O: ", vehicle_no, end="")
    
    if break_flag:
        run = False
        
    screen.fill((0, 0, 0))
    pygame.display.update()