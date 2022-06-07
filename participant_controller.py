from controller import Robot
import casadi as ca
from util import *
import param as p
import numpy as np
import networkx as nx
from math import floor
from search_based_planning_package import Planner, euclidean
from scipy import spatial
import copy
import time
from datetime import datetime


RESOLUTION = p.og_res #Global Resolution also used in search_based_planing_package
OPTIMAL_PATH = []
g_prev_yaw = None

avg_control_computation_time = 0
avg_planning_computation_time = 0

n_itr=0
state_list = []
ref_yaw_list=[]
ref_track_list = []
vel_omega_list = []
#Greedy Approach for best path calculation
def give_best_path(G, cur_position, waypoints):
	
	k = len(waypoints)
	ctr = 0

	global OPTIMAL_PATH

	def euclidean(node1, node2):
		x1,y1 = node1
		x2,y2 = node2
		return np.sqrt((x1-x2)**2 + (y1-y2)**2)

	bp = []

	while(ctr < k):
	
		tmp = []

		for p in waypoints:

			astar_path = nx.astar_path(G,cur_position,p,heuristic=euclidean,weight='cost')
			tmp.append(len(astar_path)-1)

		next_point = waypoints[tmp.index(min(tmp))]
		bp.append(next_point)
		waypoints.remove(next_point)
		cur_position = next_point
		ctr += 1

	OPTIMAL_PATH = bp

def lol(cords, cur_yaw,cur_pose):
	global RESOLUTION, target_iter, g_prev_yaw
	final_local_path = []
	tmp_yaw = 0
	prev_yaw = cur_yaw
	# final_local_path.append([cur_pose[0], cur_pose[1], prev_yaw])
	tmp_horizon = 1
	#TODO - see what happens when this removed - and this is where adaptivness horizon will come in
	if len(cords)> RESOLUTION:
		tmp_horizon = RESOLUTION

	flag_turn_or_straight = 1
	ctr_res = 0
	for i in range(len(cords)-tmp_horizon):

		tmp_yaw = np.arctan2(
			float(cords[i+1][1]-cords[i][1]), float(cords[i+1][0] - cords[i][0]))
		counter_pi = np.abs(np.abs(prev_yaw-tmp_yaw) - 2*np.pi) < 0.1

		if prev_yaw != tmp_yaw and not counter_pi and (i!=0 or (i==0 and np.abs(np.abs(prev_yaw)-np.pi) >0.8) and np.abs(np.abs(tmp_yaw)-np.pi) <0.2):
			if i!=0:
				final_local_path.append(tuple((cords[i][0]-4.5, cords[i][1]-4.5, prev_yaw)))
			flag_turn_or_straight = 0

		if flag_turn_or_straight and i != 0:
			ctr_res += 1

		final_local_path.append(tuple((cords[i][0]-4.5, cords[i][1]-4.5, tmp_yaw)))
		prev_yaw = tmp_yaw
		g_prev_yaw = prev_yaw
	

	
	if len(final_local_path) == 0:
		final_local_path.append(tuple((cords[-1][0]-4.5, cords[-1][1]-4.5, g_prev_yaw)))
	else:
		final_local_path.append(tuple((cords[-1][0]-4.5, cords[-1][1]-4.5, prev_yaw)))
	
	if p.adaptive_resolution:
		if ctr_res > 2*RESOLUTION: #If a straight segment, then reduce resolution
			RESOLUTION = p.less_res
		else:
			RESOLUTION = p.og_res
		
	return final_local_path

traverse_time_start = time.time()
robot = Robot()

timestep = int(robot.getBasicTimeStep())

TURN_SMOOTHNESS = p.turn_smoothness
CORRECTIVE_TURN_SMOOTHENING = p.corrective_turn_smoothening
ADAPTIVE_WEIGHTS = p.adaptive_weights
ADAPTIVE_HORIZON = p.adaptive_horizon


waypoints = []
waypoints_string = robot.getCustomData()
waypoints_split = waypoints_string.split()
for i in range(10):
	waypoints_element = [float(waypoints_split[2*i]), float(waypoints_split[2*i+1])]
	waypoints.append(waypoints_element)
print('Waypoints:', waypoints)

# begin{please do not change}
if (len(waypoints_split) != 20):
	waypoints_string = ' '.join(waypoints_split[:20])
# end{please do not change}

global_wp = [(x[0]+5.0, x[1]+5.0) for x in waypoints]

# Initialize devices
motor_left = robot.getDevice('wheel_left_joint')
motor_right = robot.getDevice('wheel_right_joint')
gps = robot.getDevice('gps')

arm_1 = robot.getDevice('arm_1_joint')
arm_2 = robot.getDevice('arm_2_joint')
arm_3 = robot.getDevice('arm_3_joint')
arm_4 = robot.getDevice('arm_4_joint')
arm_5 = robot.getDevice('arm_5_joint')
arm_6 = robot.getDevice('arm_6_joint')
gps_ee = robot.getDevice('gps_ee')
imu = robot.getDevice('inertial unit')
lidar = robot.getDevice('lidar_tilt')

motor_left.setVelocity(0.0)
motor_right.setVelocity(0.0)

motor_left.setPosition(float('inf'))
motor_right.setPosition(float('inf'))

gps_ee.enable(timestep)
gps.enable(timestep)
imu.enable(timestep)
lidar.enable(timestep)
lidar.enablePointCloud()

def control_arm(arm, initial_pos, final_pos, step = 1, interval = 25):
	initial_pos = int(initial_pos * interval)
	final_pos = int(final_pos * interval)
	for i in range(initial_pos, final_pos, step):
		arm.setPosition(i/interval)
		robot.step(timestep)
		gps_ee_vals = gps_ee.getValues()
		robot.setCustomData(waypoints_string + ' ' + str(gps_ee_vals[0]) + ' ' + str(gps_ee_vals[1]))


def folding_arm():
	control_arm(arm_2,0,1)
	control_arm(arm_1,1.57,2.68)
	control_arm(arm_4,0,2)

def unfolding_arm():
	control_arm(arm_4,2,0,-1)
	control_arm(arm_1,2.68,1.57,-1)
	control_arm(arm_2,1,0,-1)

def init_arm():
	control_arm(arm_1,0.07,1.57)
	control_arm(arm_3,0,-3.14,-1)

init_arm()
folding_arm()


obstacles_abs = np.array([])

global_map = np.zeros((10, 10))

start_pose = np.array([0.5, 0.5, 0.0])

waypoints_updated = []

for wp in waypoints:

	wp_new = [None, None]

	wp_new[0] = int(floor(round(wp[0]+5, 1)))
	wp_new[1] = int(floor(round(wp[1]+5, 1)))
	waypoints_updated.append(tuple(wp_new))

og_waypoints = waypoints_updated[:]

planner = Planner()

start_planner = (0, 0)  # TODO- to be changed

global_path = []
give_best_path(planner.G.copy(),start_planner,copy.deepcopy(waypoints_updated))
global_path = planner.compute_path(start_planner, OPTIMAL_PATH[0],RESOLUTION)

wp_iter = waypoints_updated.index(OPTIMAL_PATH[0])


# T = 0.007		#delta_time
N = 20			#MPC Prediction horizon

rob_diam = 1
wheel_rad = 1

v_max, v_min = 10.15, -10.15	#Constraints on velocity


if not ADAPTIVE_WEIGHTS:
	"""
	Initialize the weight matrices for MPC cost function
	"""
	Q_x = 8190			
	Q_y = 8190
	Q_theta = 9990		#x,y,theta state costs
	R1 =  20			#linear velocity actuation cost
	R2 = 190			#angular velocity actuation cost

if ADAPTIVE_WEIGHTS == False and not p.adaptive_resolution:
	#If resolution is not changed this tuning works well
	"""
	Initialize the weight matrices for MPC cost function
	"""
	Q_x = 8190			
	Q_y = 8190
	Q_theta = 9990		#x,y,theta state costs
	R1 = 7				#linear velocity actuation cost
	R2 = 190			#angular velocity actuation cost

#Turns predicted before
if not CORRECTIVE_TURN_SMOOTHENING:
	v_max, v_min = 2, -2
	"""
	Initialize the weight matrices for MPC cost function
	"""
	Q_x = 56190			
	Q_y = 56190
	Q_theta = 19990		#x,y,theta state costs
	R1 =  500				#linear velocity actuation cost
	R2 = 30		#angular velocity actuation cost



"""
Initialize the state matrix
"""
x, y, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.numel()

""""
Initialize the control matrix
"""
v, omega = ca.SX.sym('v'), ca.SX.sym('omega')
controls = ca.vertcat(v, omega)
n_controls = controls.numel()


rhs = ca.vertcat(v*ca.cos(theta), v*ca.sin(theta), omega)
f = ca.Function('f', [states, controls], [rhs])		#function to calculate the 'rhs' value, to be used for state prediction

X = ca.SX.sym('X', n_states, N+1)		#To store the predicted states
U = ca.SX.sym('U', n_controls, N)		#To store the computed controls

obj = 0

if not ADAPTIVE_WEIGHTS and not ADAPTIVE_HORIZON:
	P = ca.SX.sym('P', n_states+N*(n_states+n_controls))	#Initial Parameters

	g = X[:, 0] - P[:n_states]		#g matrix stores the constraints

	Q = ca.diagcat(Q_x, Q_y, Q_theta)
	R = ca.diagcat(R1, R2)

	step_horizon = 0.15			#for runge-kutta prediction
	for k in range(0, N):
		st, con = X[:, k], U[:, k]
		# obj = obj + (st - P[5*(k+1)-2:5*(k+1)+1]).T @ Q @ (st - P[5*(k+1)-2:5*(k+1)+1]) + (
		# 	con-P[5*(k+1)+1:5*(k+1)+3]).T @ R @ (con-P[5*(k+1)+1:5*(k+1)+3])				#The objectove function
		obj = obj + ((st - P[5*(k+1)-2: 5*(k+1)+1]).T @ Q @ (st - P[5*(k+1)-2: 5*(k+1)+1]) + (
			con-P[5*(k+1)+1: 5*(k+1)+3]).T @ R @ (con-P[5*(k+1)+1: 5*(k+1)+3]))
		# cost_pred_st = ca.mtimes(  ca.mtimes( (X[0:n_states,i] - P[n_states*(i+1) :n_states*(i+1) + n_states ].reshape((n_states,1)) ).T , Q )  ,  (X[0:n_states,i] - P[n_states*(i+1) :n_states*(i+1) + n_states ].reshape((n_states,1)) )  )  + ca.mtimes(  ca.mtimes( ( (U[0:n_controls,i]) - P[n_states*(N+1)+n_controls*(i):n_states*(N+1)+n_controls*(i) + n_controls].reshape((n_controls,1)) ).T , R )  ,  U[0:n_controls,i] - P[n_states*(N+1)+n_controls*(i):n_states*(N+1)+n_controls*(i) + n_controls].reshape((n_controls,1))  )  
		st_next = X[:, k+1]
		k1 = f(st, con)
		k2 = f(st + step_horizon/2*k1, con)
		k3 = f(st + step_horizon/2*k2, con)
		k4 = f(st + step_horizon * k3, con)
		st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
		g = ca.vertcat(g, st_next - st_next_RK4)		#Make the predicted state  

if ADAPTIVE_WEIGHTS and ADAPTIVE_HORIZON:
	P = ca.SX.sym('P', N+5+n_states+N*(n_states+n_controls))	#Initial Parameters

	g = X[:, 0] - P[N+5:N+5+n_states]		#g matrix stores the constraints

	# Q = ca.diagcat(Q_x, Q_y, Q_theta)
	# R = ca.diagcat(R1, R2)

	Q = ca.diagcat(P[N+0], P[N+1], P[N+2])
	R = ca.diagcat(P[N+3], P[N+4])
	step_horizon = 0.15			#for runge-kutta prediction
	for k in range(0, N):
		st, con = X[:, k], U[:, k]
		# obj = obj + (st - P[5*(k+1)-2:5*(k+1)+1]).T @ Q @ (st - P[5*(k+1)-2:5*(k+1)+1]) + (
		# 	con-P[5*(k+1)+1:5*(k+1)+3]).T @ R @ (con-P[5*(k+1)+1:5*(k+1)+3])				#The objectove function
		obj = obj + P[k]*((st - P[N+5+5*(k+1)-2:N+5+ 5*(k+1)+1]).T @ Q @ (st - P[N+5+5*(k+1)-2:N+5+ 5*(k+1)+1]) + (
			con-P[N+5+5*(k+1)+1:N+5+ 5*(k+1)+3]).T @ R @ (con-P[N+5+5*(k+1)+1:N+5+ 5*(k+1)+3]))
		# cost_pred_st = ca.mtimes(  ca.mtimes( (X[0:n_states,i] - P[n_states*(i+1) :n_states*(i+1) + n_states ].reshape((n_states,1)) ).T , Q )  ,  (X[0:n_states,i] - P[n_states*(i+1) :n_states*(i+1) + n_states ].reshape((n_states,1)) )  )  + ca.mtimes(  ca.mtimes( ( (U[0:n_controls,i]) - P[n_states*(N+1)+n_controls*(i):n_states*(N+1)+n_controls*(i) + n_controls].reshape((n_controls,1)) ).T , R )  ,  U[0:n_controls,i] - P[n_states*(N+1)+n_controls*(i):n_states*(N+1)+n_controls*(i) + n_controls].reshape((n_controls,1))  )  
		st_next = X[:, k+1]
		k1 = f(st, con)
		k2 = f(st + step_horizon/2*k1, con)
		k3 = f(st + step_horizon/2*k2, con)
		k4 = f(st + step_horizon * k3, con)
		st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
		g = ca.vertcat(g, st_next - st_next_RK4)		#Make the predicted state  

if (not ADAPTIVE_WEIGHTS) and ADAPTIVE_HORIZON:
	P = ca.SX.sym('P', N+n_states+N*(n_states+n_controls))	#Initial Parameters

	g = X[:, 0] - P[N:N+n_states]		#g matrix stores the constraints

	Q = ca.diagcat(Q_x, Q_y, Q_theta)
	R = ca.diagcat(R1, R2)
	
	step_horizon = 0.15			#for runge-kutta prediction
	for k in range(0, N):
		st, con = X[:, k], U[:, k]
		# obj = obj + (st - P[5*(k+1)-2:5*(k+1)+1]).T @ Q @ (st - P[5*(k+1)-2:5*(k+1)+1]) + (
		# 	con-P[5*(k+1)+1:5*(k+1)+3]).T @ R @ (con-P[5*(k+1)+1:5*(k+1)+3])				#The objectove function
		obj = obj + P[k]*((st - P[N+5*(k+1)-2:N+ 5*(k+1)+1]).T @ Q @ (st - P[N+5*(k+1)-2:N+ 5*(k+1)+1]) + (
			con-P[N+5*(k+1)+1:N+ 5*(k+1)+3]).T @ R @ (con-P[N+5*(k+1)+1:N+ 5*(k+1)+3]))
		# cost_pred_st = ca.mtimes(  ca.mtimes( (X[0:n_states,i] - P[n_states*(i+1) :n_states*(i+1) + n_states ].reshape((n_states,1)) ).T , Q )  ,  (X[0:n_states,i] - P[n_states*(i+1) :n_states*(i+1) + n_states ].reshape((n_states,1)) )  )  + ca.mtimes(  ca.mtimes( ( (U[0:n_controls,i]) - P[n_states*(N+1)+n_controls*(i):n_states*(N+1)+n_controls*(i) + n_controls].reshape((n_controls,1)) ).T , R )  ,  U[0:n_controls,i] - P[n_states*(N+1)+n_controls*(i):n_states*(N+1)+n_controls*(i) + n_controls].reshape((n_controls,1))  )  
		st_next = X[:, k+1]
		k1 = f(st, con)
		k2 = f(st + step_horizon/2*k1, con)
		k3 = f(st + step_horizon/2*k2, con)
		k4 = f(st + step_horizon * k3, con)
		st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
		g = ca.vertcat(g, st_next - st_next_RK4)	

if ADAPTIVE_WEIGHTS and (not ADAPTIVE_HORIZON):
	P = ca.SX.sym('P', 5+n_states+N*(n_states+n_controls))	#Initial Parameters

	g = X[:, 0] - P[5:5+n_states]		#g matrix stores the constraints

	# Q = ca.diagcat(Q_x, Q_y, Q_theta)
	# R = ca.diagcat(R1, R2)

	Q = ca.diagcat(P[0], P[1], P[2])
	R = ca.diagcat(P[3], P[4])
	step_horizon = 0.15			#for runge-kutta prediction
	for k in range(0, N):
		st, con = X[:, k], U[:, k]
		# obj = obj + (st - P[5*(k+1)-2:5*(k+1)+1]).T @ Q @ (st - P[5*(k+1)-2:5*(k+1)+1]) + (
		# 	con-P[5*(k+1)+1:5*(k+1)+3]).T @ R @ (con-P[5*(k+1)+1:5*(k+1)+3])				#The objectove function
		obj = obj + ((st - P[5+5*(k+1)-2:5+ 5*(k+1)+1]).T @ Q @ (st - P[5+5*(k+1)-2:5+ 5*(k+1)+1]) + (
			con-P[5+5*(k+1)+1:5+ 5*(k+1)+3]).T @ R @ (con-P[5+5*(k+1)+1:5+ 5*(k+1)+3]))
		# cost_pred_st = ca.mtimes(  ca.mtimes( (X[0:n_states,i] - P[n_states*(i+1) :n_states*(i+1) + n_states ].reshape((n_states,1)) ).T , Q )  ,  (X[0:n_states,i] - P[n_states*(i+1) :n_states*(i+1) + n_states ].reshape((n_states,1)) )  )  + ca.mtimes(  ca.mtimes( ( (U[0:n_controls,i]) - P[n_states*(N+1)+n_controls*(i):n_states*(N+1)+n_controls*(i) + n_controls].reshape((n_controls,1)) ).T , R )  ,  U[0:n_controls,i] - P[n_states*(N+1)+n_controls*(i):n_states*(N+1)+n_controls*(i) + n_controls].reshape((n_controls,1))  )  
		st_next = X[:, k+1]
		k1 = f(st, con)
		k2 = f(st + step_horizon/2*k1, con)
		k3 = f(st + step_horizon/2*k2, con)
		k4 = f(st + step_horizon * k3, con)
		st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
		g = ca.vertcat(g, st_next - st_next_RK4)			



OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))		#Combine the X and U matrices for multi-shooting implementation of MPC

"""
Initialize the nlp problem
"""
nlp_prob = {
	'f': obj,
	'x': OPT_variables,
	'g': g,
	'p': P
}

opts = {
	'ipopt': {
		'max_iter': 2000,
		'print_level': 0,
		'acceptable_tol': 1e-8,
		'acceptable_obj_change_tol': 1e-6
	},
	'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

"""
Add the lower and upper bounds
"""
lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = -ca.inf     # X lower bound
lbx[1: n_states*(N+1): n_states] = -ca.inf     # Y lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

ubx[0: n_states*(N+1): n_states] = ca.inf     # X upper bound
ubx[1: n_states*(N+1): n_states] = ca.inf      # Y upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound

lbx[n_states*(N+1):] = v_min                  # v lower bound for all V
ubx[n_states*(N+1):] = v_max                  # v upper bound for all V



args = {
		# constraints lower bound
		'lbg': ca.DM.zeros((n_states*(N+1), 1)),
		# constraints upper bound
		'ubg': ca.DM.zeros((n_states*(N+1), 1)),
		'lbx': lbx,
		'ubx': ubx
	}


u0 = ca.DM.zeros((n_controls, N))		#Initialize an empty control matrix

def apply_mpc(u):		#Function for lower level control of the bot
	global vel_omega_list
	u_app = u[:, 0]
	v, w = float(u_app[0, 0].full()[0, 0]), float(u_app[1, 0].full()[0, 0])
	vel_omega_list.append((v,w))
	vel_left = (v-w*rob_diam)/wheel_rad
	vel_right = (v+w*rob_diam)/wheel_rad
	"""
	To ensure the velocity limit
	"""
	if vel_right > v_max:
		vel_right = v_max
	if vel_right<v_min:
		vel_right = v_min
	if vel_left > v_max:
		vel_left = v_max
	if vel_left < v_min:
		vel_left = v_min
	motor_left.setVelocity(vel_left)
	motor_right.setVelocity(vel_right)
	"""
	Instead of initializing the control matrices with zero, initialize it with MPC's computed control actions to reach at a solution faster
	"""
	u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))
	return u0

X0 = ca.repmat(ca.DM([0, 0, 0]), 1, N+1)

# mpciter = 0
target_iter = 0

first_ever = True
first_ever_obstacle = False

obstacle_found = False
global_map_tmp = np.zeros((10, 10))
counter_bp = True
counter_stuck = 0		#+59, needs tuning
counter_try = 1
counter_main =  1
counter_jerk = 0
# model_error =[]
counter_stuck2 = 0
counter_yaw_error = 0

while robot.step(timestep) != -1:
	# print(R2)
	# print(P[0])
	# print("wp_iter : ",wp_iter)
	# print("wp_upd : ",waypoints_updated)
	# print("Q_T",Q_theta)
	# print("RESO", RESOLUTION)
	# print("ITER", wp_iter)
	# print("WP_UPD", waypoints_updated)
	counter_jerk += 1
	gps_ee_vals = gps_ee.getValues()
	robot.setCustomData(waypoints_string + ' ' + str(gps_ee_vals[0]) + ' ' + str(gps_ee_vals[1]))

	imu_rads = imu.getRollPitchYaw()
	orientation = np.array([x for x in imu_rads])
	tmp_yaw = (-np.pi/2-orientation[0])
	if orientation[0] < np.pi/2:
		yaw = -1*(-np.pi/2-orientation[0])
	else:
		yaw = orientation[0]-3*np.pi/2

	jerk = False
	if(np.abs(orientation[1]) > 0.01 or np.abs(np.abs(orientation[2]) - np.pi/2) > 0.01):
		jerk = True
	transform_matrix = np.array(
		[[np.cos(tmp_yaw), np.sin(tmp_yaw)], [-np.sin(tmp_yaw), np.cos(tmp_yaw)]])

	pc = lidar.getPointCloud()
	t = lidar.getRangeImage()

	x_cur, y_cur, theta_cur = gps.getValues()[0], gps.getValues()[1], yaw

	car_pose = np.array([x_cur+5.0, y_cur+5.0])
	cur_node = (int(x_cur+5), int(y_cur+5))

	
	if cur_node in waypoints_updated:
		try:
			wp_cur_index = waypoints_updated.index(cur_node)
			if wp_cur_index != wp_iter and wp_cur_index != len(waypoints_updated)-1:
				distance = np.hypot(gps_ee_vals[0]+4.5-waypoints_updated[wp_cur_index][0], gps_ee_vals[1]+4.5-waypoints_updated[wp_cur_index][1]) 
	
				if distance < 0.13:
					wp_element = waypoints_updated[wp_cur_index]
					waypoints_updated.pop(wp_cur_index)
					
					for i in OPTIMAL_PATH:
						if i not in waypoints_updated:
							OPTIMAL_PATH.remove(i)
							
					
					if len(waypoints_updated) == 0:
						break

					wp_iter = waypoints_updated.index(OPTIMAL_PATH[0])
		except:
			print("Index not found")

	votes = np.zeros((10, 10))

	if(not jerk) and counter_jerk>40:
		for idx, tmp in enumerate(pc):
			transformed_point = np.array([-tmp.z+0.202, -tmp.x])

			if(transformed_point[0] != float('inf') and transformed_point[0] != float('-inf') and ~np.isnan(transformed_point[0])):
				if(transformed_point[1] != float('inf') and transformed_point[1] != float('-inf') and ~np.isnan(transformed_point[1])):
					if ((idx > 31) and (idx < 635)):
						if np.hypot(transformed_point[0],transformed_point[1])<4:
							x_offset, y_offset = set_offset(tmp_yaw, idx)
							transformed_point = (
								transform_matrix@transformed_point+car_pose)
							transformed_point[0] = int(
								floor(transformed_point[0]+x_offset))
							transformed_point[1] = int(
								floor(transformed_point[1]+y_offset))

							if (0 <= transformed_point[0] < 10) and (0 <= transformed_point[1] < 10):
								votes[int(transformed_point[0]), int(
									transformed_point[1])] += 1

		# print("votes", votes)
		votes[votes <= 65] = 0
		votes[votes > 65] = 1

		for wp in og_waypoints:
			if euclidean(wp,(x_cur+4.5,y_cur+4.5))<0.85:
				votes[int(wp[0]), int(wp[1])] = 0

	global_map = or_gate(global_map, votes)
	global_map_diff = global_map_tmp - global_map
	# print(global_map)
	global_map_tmp[:] = global_map

	# print(global_map_diff)
	if np.sum(global_map_diff) != 0:
		obstacle_found = True
		if first_ever:
			first_ever_obstacle = True

	first_ever = False
	planner.grid = global_map
	planner.delete_node()
	
	if counter_main:
		cur_node = (int(x_cur+5), int(y_cur+5))
		give_best_path(planner.G.copy(),cur_node,copy.deepcopy(waypoints_updated))
		computation_time_planning_start = time.time()
		wp_iter = waypoints_updated.index(OPTIMAL_PATH[0])
		try:
			global_path_tmp = planner.compute_path(
				cur_node, waypoints_updated[wp_iter],RESOLUTION)
			counter_try = 1


		except:
			print(f"Path not found to the current targeted waypoint - {waypoints_updated[wp_iter]}, skipping ahead")
			wp_element = waypoints_updated[wp_iter]
			counter_try += 1
			if len(OPTIMAL_PATH) > counter_try:
				wp_iter = waypoints_updated.index(OPTIMAL_PATH[counter_try])	
			else:
				counter_try = 1
				first_ever_obstacle = True
				print("NO WAYPOINTS left to navigate , exiting the code")
				# waypoints_updated.append(og_waypoints[-1])
			continue
		
		
		if global_path_tmp != global_path:
			target_iter = 0

		global_path = global_path_tmp

		data = np.array(lol(global_path, yaw,(x_cur,y_cur)))

		target_list = data
		

		if len(target_list)>0:
			#Handling the yaw discontinuity at 2pi
			if (np.abs(target_list[0][2]-np.pi) < 0.2) and yaw < 0:
				yaw += 2*np.pi
			if (np.abs(target_list[0][2]+np.pi) < 0.2) and yaw > 0:
				yaw -= 2*np.pi

			#Checking shorter(clockwise or anticlowise)turn
			if (np.abs(yaw+np.pi) < 0.04) and (target_list[0][2] >= 0):
				yaw = -1*yaw
			if (np.abs(yaw-np.pi) < 0.04) and (target_list[0][2] < 0):
				yaw = -1*yaw
			if yaw>0 and target_list[0][2] <0:
				if np.abs(yaw -target_list[0][2]) >np.pi:
					target_list[0][2] += 2*np.pi
			if target_list[0][2] > np.pi and yaw<0:
				yaw+=2*np.pi
				
			if yaw<0 and target_list[0][2] >0:
				if np.abs(yaw -target_list[0][2]) >np.pi:
					target_list[0][2] -= 2*np.pi
			if target_list[0][2] < -np.pi and yaw>0:
				yaw-=2*np.pi

			if (np.abs(target_list[0][2]-3*np.pi/2) < 0.2) and yaw > 0:
				target_list[0][2] = 2*np.pi-target_list[0][2]
			theta_cur = yaw

		if p.adaptive_resolution:
			slice_index = -1		#To reduce the resolution of the path if it is a straight linr, leads to faster speeds in MPC
			if len(target_list) >0 and np.abs(np.abs(np.abs(theta_cur) - np.abs(target_list[0][2]))) >0.2:	#Checks whether or not the bot deviates too much from the straight line to apply slicing
				slice_index =  0
			if slice_index == -1:
				for i in range(len(target_list)-1):
					if np.abs(np.abs(np.abs(target_list[i+1][2]) - np.abs(target_list[i][2]))) >0.2:
						break 
					slice_index = i
			if slice_index != -1:
				target_list = list(target_list)
				target_list = target_list[:slice_index:2]+ target_list[slice_index:] #TODO try tpo make the skip counter adaptive
				target_list = np.array(target_list)


		# print("target_iter = ", target_iter, "target_list = ", target_list)
		#TODO - implement corrected turn smoothness by adaptive horizon
		if CORRECTIVE_TURN_SMOOTHENING == True:
			for i in range(len(target_list)-1):		#Don't give the waypoints after the turn untill very near to the predicted turning point
				# print(TURN_SMOOTHNESS)
				if np.abs(np.abs(np.abs(target_list[i+1][2]) - np.abs(target_list[i][2]))-np.pi/2) <0.2 and (np.hypot(x_cur-target_list[i][0],y_cur -target_list[i][1]) >TURN_SMOOTHNESS):
					# print("ENTERED")
					target_list = target_list[:i]
					break
		# print("target_iter = ", target_iter, "target_list = ", target_list)
		data_new = []
		for i in target_list:
			data_new.append(tuple(i))

		distance,index = spatial.KDTree(data_new).query((x_cur,y_cur,theta_cur))


		if (np.abs(target_list[index][2]-np.pi) < 0.2) and yaw < 0:
			yaw += 2*np.pi
		if (np.abs(target_list[index][2]+np.pi) < 0.2) and yaw > 0:
			yaw -= 2*np.pi

		if (np.abs(yaw+np.pi) < 0.04) and (target_list[index][2] >= 0):
			yaw = -1*yaw
		if (np.abs(yaw-np.pi) < 0.04) and (target_list[index][2] < 0):
			yaw = -1*yaw
		#Checking shorter(clockwise or anticlowise)turn
		if yaw>0 and target_list[index][2] <0:
			if np.abs(yaw -target_list[index][2]) >np.pi:
				target_list[index][2] += 2*np.pi
		if target_list[index][2] > np.pi and yaw<0:
			yaw+=2*np.pi
			
		if yaw<0 and target_list[index][2] >0:
			if np.abs(yaw -target_list[index][2]) >np.pi:
				target_list[index][2] -= 2*np.pi
		if target_list[index][2] < -np.pi and yaw>0:
			yaw-=2*np.pi

		# print("TRAGET_LIST", target_list)
		if (np.abs(np.abs(target_list[index][2])-3*np.pi/2) < 0.2) and yaw > 0:
			target_list[index][2] = 2*np.pi-target_list[index][2]
		theta_cur = yaw
		# print("CUR_THETA", theta_cur)
		computation_time_planning_end = time.time() - computation_time_planning_start

		state_list.append((x_cur+4.5, y_cur+4.5, theta_cur))
		ref_track_list.append(cur_node)
		ref_yaw_list.append(target_list[index][2])

		computation_time_control_start = time.time()
		if ADAPTIVE_WEIGHTS:
			"""
			Adaptive Tuning
			"""
			Q_x = 9190		
			Q_y = 9190
			Q_theta = 11990		#x,y,theta state costs
			R1 = 70	#linear velocity actuation cost
			R2 =1100
			if np.abs(theta_cur-target_list[index][2]) < 0.06:  #If no error in yaw
				counter_yaw_error +=1
				if counter_yaw_error > 90:
					# print("LETS GOOOOO")
					# wheel_rad= 0.6
					Q_x = 7190			
					Q_y = 7190
					Q_theta = 9990		#x,y,theta state costs
					R1 = 0.001		#linear velocity actuation cost
					R2 = 40
			else:
				counter_yaw_error = 0
				# wheel_rad = 1.05
				Q_x = 9190			
				Q_y = 9190
				Q_theta = 11990		#x,y,theta state costs
				R1 = 70			#linear velocity actuation cost
				R2 = 600

			
			adap_weights = ca.DM([Q_x, Q_y,Q_theta,R1,R2])
		x_0 = ca.DM([x_cur, y_cur, theta_cur])		#initial state
		p_matrix = []
		if ADAPTIVE_HORIZON:
			"""
			Adaptive prediction horizon
			"""
			if len(target_list) < 20:
				N_new = len(target_list) + 3
			N_matrix = []
			for i in range(N):
				if i <N_new:
					N_matrix.append(1)
				else:
					N_matrix.append(0)

			p_matrix = ca.vertcat(p_matrix,N_matrix)
			# print(N_new,p_matrix)
		
		if ADAPTIVE_WEIGHTS:
			"""
			Params for Adaptive tuning in p_matrix (P matrix)
			"""
			p_matrix = ca.vertcat(p_matrix,adap_weights)

		p_matrix = ca.vertcat(p_matrix,x_0)
		# print(p_matrix)
		
		for k in range(0,N):
			if (index+1)<len(target_list):
				x_ref,y_ref = target_list[index][0],target_list[index][1]
				theta_ref = target_list[index][2]
				u_ref, omega_ref = 10.15,0
				#print("woweee =", theta_ref)
			else:
				x_ref,y_ref = target_list[-1][0],target_list[-1][1]
				theta_ref = target_list[-1][2]
				u_ref, omega_ref = -1.5,0
				# print("WOOOO")
	
			index+=1
			p_matrix = ca.vertcat(p_matrix,x_ref,y_ref,theta_ref)
			p_matrix = ca.vertcat(p_matrix,u_ref,omega_ref)
			
		# print("error = ", distance)


		args['p'] = ca.vertcat(p_matrix)
		args['x0'] = ca.vertcat(ca.reshape(X0, n_states*(N+1), 1),ca.reshape(u0, n_controls*N, 1))
		
		sol = solver(x0=args['x0'],lbx=args['lbx'],ubx=args['ubx'],lbg=args['lbg'],ubg=args['ubg'],p=args['p'])
		
		u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
		# u = ca.reshape(sol['x'].T, n_controls, N) 
		X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

		X0 = ca.horzcat(
			X0[:, 1:],
			ca.reshape(X0[:, -1], -1, 1)
		)

		u0 = apply_mpc(u)
		computation_time_control_end = time.time() - computation_time_control_start
		avg_control_computation_time = (avg_control_computation_time*n_itr +  computation_time_control_end) / (n_itr+1)
		avg_planning_computation_time = (avg_planning_computation_time*n_itr +  computation_time_planning_end) / (n_itr+1)
		n_itr+=1
		# print("COSTS", R2)

		# mpciter += 1
		error = ((x_cur-target_list[-1][0])**2+(y_cur -target_list[-1][1])**2+(theta_cur-target_list[-1][2])**2)**0.5
		error_yaw = np.abs(theta_cur - target_list[-1][2])

		# if error>0.15 and error<0.2:
		# 	if error_yaw < 0.07:
		# 		error = 0.14
		# 		motor_left.setVelocity(0)
		# 		motor_right.setVelocity(0)

		# print("error = ", error)
		# print("ERROR_YAW", error_yaw)

		if error>0.15 and error<0.35:
			if error_yaw < 0.2:
				counter_stuck2 += 1
				if counter_stuck2 == 50:
					# print("RECOVERY_NEW")
					counter_stuck2 = 0
					# motor_left.setVelocity(0)
					# motor_right.setVelocity(0)
					# unfolding_arm()
					# if not (wp_iter == len(waypoints_updated)-1 and len(waypoints_updated) == 1):
					# 	folding_arm()
					counter_main = 0
			else:
				counter_stuck2 = 0
		else:
			counter_stuck2 = 0
		
		# print("CS", counter_stuck2)

		if np.abs(motor_left.getVelocity()) < 1e-1 and np.abs(motor_right.getVelocity()) < 1e-1 and not first_ever:
			# if (car_pose[0] !=target_list[0][0]) or (car_pose[1] != target_list[0][1]):
			counter_stuck += 1
		if counter_stuck == 10:
			
			# print("Entered Recovery")
			counter_stuck = 0
			if error>0.15 and error<0.2:
				if error_yaw < 0.07:
					error = 0.14
					motor_left.setVelocity(0)
					motor_right.setVelocity(0)

			if cur_node in waypoints_updated:
				try:
					wp_cur_index = waypoints_updated.index(cur_node)
					if wp_cur_index != len(waypoints_updated)-1:
						distance = np.hypot(car_pose[0]-0.5-waypoints_updated[wp_cur_index][0], car_pose[1]-0.5-waypoints_updated[wp_cur_index][1]) 
						
						if distance < 0.13:
							wp_element = waypoints_updated[wp_cur_index]
							waypoints_updated.pop(wp_cur_index)
							
							for i in OPTIMAL_PATH:
								if i not in waypoints_updated:
									OPTIMAL_PATH.remove(i)
									
							
							if len(waypoints_updated) == 0:
								break

							wp_iter = waypoints_updated.index(OPTIMAL_PATH[0])
				except:
					print("Index not found")
			else:
				k = 0
				while(robot.step(timestep) != -1):
					k+=1
					motor_left.setVelocity(1.5)
					motor_right.setVelocity(1.3)
					if k>20:
						break

		if np.absolute(error) < 0.16:
			# motor_left.setVelocity(0)
			# motor_right.setVelocity(0)
		# 	unfolding_arm()
		# 	if not (wp_iter == len(waypoints_updated)-1 and len(waypoints_updated) == 1):
		# 		folding_arm()
		# 		# print("GPS_EE", gps_ee_vals[0], gps_ee_vals[1])
			counter_main = 0
		# 	#print("target_iter +1")
		# 	target_iter += 1
	
		# print(f"motor left = {motor_left.getVelocity()}, motor right = {motor_right.getVelocity()}")
		# print("CS", counter_stuck)
		# print("FE", first_ever)
		# print("LEFT", motor_left.getVelocity())
		# print("RIGHT", motor_right.getVelocity())


	else:
		counter_jerk = 0
		# counter_stuck2 = 0
		counter_main = 1
		wp_element = waypoints_updated[wp_iter]
		waypoints_updated.pop(wp_iter)
		
		for i in OPTIMAL_PATH:
			if i not in waypoints_updated:
				OPTIMAL_PATH.remove(i)
				
		
		if len(waypoints_updated) == 0 :
			break 
		wp_iter = waypoints_updated.index(OPTIMAL_PATH[0])
		print('wp_iter_now = ', wp_iter)
		motor_left.setVelocity(0)
		motor_right.setVelocity(0)

	pass

traverse_time_end = traverse_time_start - time.time()
# avg_control_computation_time , avg_planning_computation_time , traverse_time_end
# state_list , ref_yaw_list , ref_track_list, vel_omega_list

print("Control computation time : ", avg_control_computation_time)
print("Planning computation time : ", avg_planning_computation_time)
print("Traversal Time: ", traverse_time_end)
print("State list : ", state_list[:100:10]) 
print("Ref Yaw list : ", ref_yaw_list[:100:10]) 
print("Ref track list : ", ref_track_list[:100:10]) 
print("Vel Omega : ", vel_omega_list[:100:10]) 

if p.logging:
	import os
	now = datetime.now()
	log_dir_path = os.path.join(
				"./logs/", f'{now.strftime("%m - %d - %Y ,  %H:%M:%S")}')
	try:
		print(log_dir_path)
		os.makedirs(log_dir_path)
	except Exception as e:
		print (e)
		print(f'Log folder already Exists at {log_dir_path} .Log files will be added there')

	time_file_path = os.path.join(log_dir_path, f'time.log')
	time_file = open(time_file_path, "a")

	state_ref_control_file_path = os.path.join(log_dir_path, f'state_ref_control.csv')
	# state_ref_control_file = open(state_ref_control_file_path, "a")

	import csv

	cols = ["X_cur", "Y_cur", "Yaw_cur", "X_ref", "Y_ref", "Yaw_ref", "vel", "omega"]

	rows = []
	for i in range(len(state_list)):
		rows.append([state_list[i][0],state_list[i][1],state_list[i][2],ref_track_list[i][0],ref_track_list[i][1],ref_yaw_list[i],vel_omega_list[i][0], vel_omega_list[i][1]])


	with time_file as f:
		f.write(f"Control computation time : {avg_control_computation_time}\n")
		f.write(f"Planning computation time : {avg_planning_computation_time}\n")
		f.write(f"Traversal Time: {traverse_time_end}")


	with open(state_ref_control_file_path, 'w') as f:

		# using csv.writer method from CSV package
		write = csv.writer(f)

		write.writerow(cols)
		write.writerows(rows)