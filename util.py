import numpy as np

def or_gate(a, b):
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			a[i, j] = a[i, j] or b[i, j]
	return a

def set_offset(yaw, ray_no, threshold=0.09):
	ray_no = ray_no-667//2
	angle = yaw+ray_no*0.36*np.pi/180+2*np.pi  # dectorad
	angle = angle % (2*np.pi)

	x_offset = 0.0
	y_offset = 0.0

	if(0 <= angle <= np.pi/2):
		x_offset = threshold
		y_offset = -threshold
	elif(np.pi/2 <= angle <= np.pi):
		x_offset = -threshold
		y_offset = -threshold
	elif(np.pi <= angle <= 3*np.pi/2):
		x_offset = -threshold
		y_offset = threshold
	else:
		x_offset = threshold
		y_offset = threshold

	return x_offset, y_offset