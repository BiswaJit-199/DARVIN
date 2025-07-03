# Utilities for head pose calculation using solvePnP
import numpy as np
# 3D model points (approximate face model in mm)
model_points = np.array([
	(0.0, 0.0, 0.0),			# Nose tip
	(0.0, -330.0, -65.0),		# Chin
	(-225.0, 170.0, -135.0),	# Left eye corner
	(225.0, 170.0, -135.0),		# Right eye corner
	(-150.0, -150.0, -125.0),	# Left mouth corner
	(150.0, -150.0, -125.0)		# Right mouth corner
])