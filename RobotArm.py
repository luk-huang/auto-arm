from xarm.wrapper import XArmAPI
import sys
import math
import os
from configparser import ConfigParser

class RobotArm():
    def __init__(self, ip):
        """
        Initialize the RobotArm object.

        This method reads the IP address from the command line argument or from
        the file '../robot.conf'. It then creates an XArmAPI object with this
        IP address and enables the motion. The state is set to 0.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
        if len(sys.argv) >= 2:
            self.ip = sys.argv[1]
        else:
            try:
                from configparser import ConfigParser
                parser = ConfigParser()
                parser.read('../robot.conf')
                self.ip = parser.get('xArm', 'ip')
            except:
                self.ip = input('Please input the xArm ip address:')
                if not ip:
                    print('input error, exit')
                    sys.exit(1)

        self.arm = XArmAPI(self.ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

    def start(self):
        """
        Initializes the robot arm by setting the counter and setting the initial angle of the gripper.

        Returns
        -------
        None
        """
        try:
            # Install xArm Gripper
            print("hither")
            code = self.arm.set_counter_reset()
            print("trying")
            weight = 0.610 
            center_of_gravity = (0.06125, 0.0458, 0.0375) 
            self.arm.set_tcp_load(weight=weight, center_of_gravity=center_of_gravity)
            code=self.arm.set_servo_angle(angle=[180,75,-180,20,0,90,-60],is_radian=False,speed=30,wait=True)

        except Exception as e:
            print('MainException: {}'.format(e))
    
    def reset_arm_counters(self):
        # Resetting the operation or movement counter of the arm
        code = self.arm.set_counter_reset()
        if code == 0:
            print("Counter reset successfully.")
        else:
            print(f"Failed to reset counter, error code: {code}")

    def moveArmTo(self, coor):
        highcoor=coor[:2]+[400]+coor[3:]
        x=coor[0]
        y=coor[1]
        new_angle=math.atan2(y,x)/math.pi*180
        new_angle+=180

  
        code = self.arm.set_servo_angle(servo_id=1,wait=True,angle=new_angle,is_radian=False,speed=100)
        code = self.arm.set_position_aa(highcoor,is_radian=False, speed=100,  mvacc=100, wait=True)
        code = self.arm.set_position_aa(coor,is_radian=False, speed=80,  mvacc=100, wait=True)

        self.arm.set_vacuum_gripper(on=False)
        self.arm.set_tcp_load(weight=0.61, center_of_gravity=(0.06125, 0.0458, 0.0375))
        code = self.arm.set_position_aa(highcoor,is_radian=False, speed=100,  mvacc=100, wait=True)
        code = self.arm.set_servo_angle(angle=[180,75,-180,20,0,90,-60],speed=100,is_radian=False,wait=True)
        return
