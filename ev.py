import numpy as np
import math
from scipy.integrate import odeint
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.state import PMState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from utilities import project_state_kb, integrate_nonlin_dynamics

"""""
1个类（4个函数），EV类：接受“初始状态”，“车辆参数”，“控制器”，“时间间隔”等4个输入，创建一个EV对象，并通过run_step函数更新车辆状态
1.__init__: EV类的构造函数，用于初始化车辆的相关属性；将输入的初始状态initial_state通过project_state_kb函数转换为[s, d, phi, v].T的形式，并赋值给_x0。
2.run_step： 计算u，并且综合车辆模型递推下一步状态，并添加至xx list和state_list
3.get_prediction：调用commonroad包预测本车轨迹？
4.generatePMstate：生成PMState对象，这种格式符合 CommonRoad 库的接口规范，便于与 CommonRoad 库中的其他功能模块进行交互和集成
"""

class EV:

    vmax = 35.0
    vmin = 0.0
    umax = np.array([5.0, 0.4])
    umin = np.array([-9.0, -0.4])
    dumax = np.array([9.0, 0.4])

    def __init__(self, initial_state, vehicle_params, controller, T):
        self._x0 = project_state_kb(initial_state) #    [s, d, phi, v].T
        self._x0_curv = False   #   curved ev state (updated externally)
        self._xx = [self._x0]   #   存储车辆的状态序列
        self._initial_state = EV.generatePMstate(self._x0, 0)
        self._state_list = [self._initial_state] #是 CommonRoad 库中用于描述车辆状态的一种对象形式
        self._shape = Rectangle(length=vehicle_params.l, width=vehicle_params.w)
        self._lr = vehicle_params.l/2
        self._lflr_ratio = 0.5
        self._controller = controller
        self._T = T

    def run_step(self, k, x0_tv, xrefo, lane_info, comparison=False):
        # Controller update in curvilinear coordinates 计算控制器输入是在曲线坐标系下进行的
        if comparison:
            u0 = self._controller.run_step_comparison(self._x0_curv, x0_tv, xrefo, lane_info, lane_info['d_max'],
                     lane_info['d_min'])
        else:
            u0 = self._controller.run_step(self._x0_curv, x0_tv, xrefo, lane_info, lane_info['d_max'],
                     lane_info['d_min'])
        #   state update takes place in cartesian coordinates，笛卡尔坐标
        self._x0 = integrate_nonlin_dynamics(self._x0, u0, self._T, self._lr, self._lflr_ratio) #状态更新在笛卡尔坐标系下进行的
        self._xx.append(self._x0)
        self._state_list.append(EV.generatePMstate(self._x0, k))
        self._x0_curv = False

    def get_prediction(self):#画过吗，在结果的展示动画上好像没有画过本车预测轨迹啊
        return TrajectoryPrediction(trajectory=Trajectory(initial_time_step=0, state_list=self._state_list),
                                                            shape=self._shape)
        
    def generatePMstate(x0, k): #PMState 对象，该对象表示车辆在某一时刻的状态，包括位置、时间步、速度和横向速度
        return PMState(**{'position': np.array([x0[0], x0[1]]), 'time_step': k,
                            'velocity': x0[3]*math.cos(x0[2]),
                            'velocity_y': x0[3]*math.sin(x0[2])})