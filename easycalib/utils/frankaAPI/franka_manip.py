from __future__ import print_function
import sys
import numpy as np
from easycalib.utils.setup_logger import setup_logger
logger = setup_logger(__name__)

DEPENDENCIES_INSTALLED = True
try:
	import rospy
	import moveit_commander
	import moveit_msgs.msg
	import geometry_msgs.msg
	from std_msgs.msg import String
	from moveit_commander.conversions import pose_to_list
except ImportError:
	logger.warning("DEPENDENCIESW are not completely installed. Required: rospy, moveit_commander, moveit_msgs.msg, geometry_msgs, std_msgs.")
	DEPENDENCIES_INSTALLED = False

import cv2
import os
import numpy as np

np.random.seed(2024)
from easycalib.utils.utilities import calculate_camera_intrinsics


try:
	from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
	from math import pi, fabs, cos, sqrt

	tau = 2.0 * pi

	def dist(p, q):
		return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

# END_SUB_TUTORIAL

REST_POSE = [
	5.928617003472516e-05,
	-0.7848036409260933,
	-0.000308854746172659,
	-2.357726806912310,
	-0.00011798564528483742,
	1.570464383098814,
	0.7852387161304554,
]

def all_close(goal, actual, tolerance):
	"""
	Convenience method for testing if the values in two lists are within a tolerance of each other.
	For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
	between the identical orientations q and -q is calculated correctly).
	@param: goal       A list of floats, a Pose or a PoseStamped
	@param: actual     A list of floats, a Pose or a PoseStamped
	@param: tolerance  A float
	@returns: bool
	"""
	if type(goal) is list:
		for index in range(len(goal)):
			if abs(actual[index] - goal[index]) > tolerance:
				return False

	elif type(goal) is geometry_msgs.msg.PoseStamped:
		return all_close(goal.pose, actual.pose, tolerance)

	elif type(goal) is geometry_msgs.msg.Pose:
		x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
		x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
		# Euclidean distance
		d = dist((x1, y1, z1), (x0, y0, z0))
		# phi = angle between orientations
		cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
		return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

	return True


class MoveGroupPythonInterface(object):
	"""MoveGroupPythonInterfaceTutorial"""

	def __init__(self):
		super(MoveGroupPythonInterface, self).__init__()

		self.dry_run = False
		self.rest_pose = REST_POSE
		if DEPENDENCIES_INSTALLED:
			# BEGIN_SUB_TUTORIAL setup
			##
			# First initialize `moveit_commander`_ and a `rospy`_ node:
			moveit_commander.roscpp_initialize(sys.argv)
			rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

			# Instantiate a `RobotCommander`_ object. Provides information such as the robot's
			# kinematic model and the robot's current joint states
			robot = moveit_commander.RobotCommander()

			# Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
			# for getting, setting, and updating the robot's internal understanding of the
			# surrounding world:
			scene = moveit_commander.PlanningSceneInterface()

			# Instantiate a `MoveGroupCommander`_ object.  This object is an interface
			# to a planning group (group of joints).  In this tutorial the group is the primary
			# arm joints in the Panda robot, so we set the group's name to "panda_arm".
			# If you are using a different robot, change this value to the name of your robot
			# arm planning group.
			# This interface can be used to plan and execute motions:
			group_name = "panda_arm"
			move_group = moveit_commander.MoveGroupCommander(group_name)

			# Create a `DisplayTrajectory`_ ROS publisher which is used to display
			# trajectories in Rviz:
			display_trajectory_publisher = rospy.Publisher(
				"/move_group/display_planned_path",
				moveit_msgs.msg.DisplayTrajectory,
				queue_size=20,
			)

			# END_SUB_TUTORIAL

			# BEGIN_SUB_TUTORIAL basic_info
			##
			# Getting Basic Information
			# ^^^^^^^^^^^^^^^^^^^^^^^^^
			# We can get the name of the reference frame for this robot:
			planning_frame = move_group.get_planning_frame()
			logger.info("============ Planning frame: %s" % planning_frame)

			# We can also print the name of the end-effector link for this group:
			eef_link = move_group.get_end_effector_link()
			logger.info("============ End effector link: %s" % eef_link)

			# We can get a list of all the groups in the robot:
			group_names = robot.get_group_names()
			logger.info("============ Available Planning Groups:", robot.get_group_names())

			# Sometimes for debugging it is useful to print the entire state of the
			# robot:
			logger.info("============ Printing robot state")
			logger.info(robot.get_current_state())
			logger.info("")

			# Sometimes it is useful to set the end-effector of panda to a virtual link, and moveit_commander uses this virtual link as a planning interface.
			logger.info("============Setting virtual link as the end effector==================")
			move_group = robot.get_group("panda_arm")
			move_group.set_end_effector_link("panda_grasptarget")
			# END_SUB_TUTORIAL

			# Misc variables
			self.box_name = ""
			self.robot = robot
			self.scene = scene
			self.move_group = move_group
			self.display_trajectory_publisher = display_trajectory_publisher
			self.planning_frame = planning_frame
			self.eef_link = eef_link
			self.group_names = group_names
		else:
			self.dry_run = True


	def go_to_rest_pose(self):
		"""
		Set the robot to the rest pose
		"""
		if self.dry_run:
			return True
		# Copy class variables to local variables to make the web tutorials more clear.
		# In practice, you should use the class variables directly unless you have a good
		# reason not to.
		move_group = self.move_group
		# The go command can be called with joint values, poses, or without any
		# parameters if you have already set the pose or joint target for the group
		move_group.go(self.rest_pose, wait=True)

		# Calling ``stop()`` ensures that there is no residual movement
		move_group.stop()

		# END_SUB_TUTORIAL

		# For testing:
		current_joints = move_group.get_current_joint_values()
		return all_close(self.rest_pose, current_joints, 0.01)

	def go_to_pose(self, joint_goal):
		"""
		Set the robot to the rest pose
		"""
		if self.dry_run:
			return True
		# Copy class variables to local variables to make the web tutorials more clear.
		# In practice, you should use the class variables directly unless you have a good
		# reason not to.
		move_group = self.move_group
		# The go command can be called with joint values, poses, or without any
		# parameters if you have already set the pose or joint target for the group
		move_group.go(joint_goal, wait=True)

		# Calling ``stop()`` ensures that there is no residual movement
		move_group.stop()

		# END_SUB_TUTORIAL

		# For testing:
		current_joints = move_group.get_current_joint_values()
		return all_close(joint_goal, current_joints, 0.01)

	def set_servo_angle(self, servo_id=None, angle=0, is_radian=True, wait=True):
		"""
		Set the angle of the specific servo_id
		"""
		if self.dry_run:
			return True
		# logger.info("The robot 's {servo_id} is moved to {angle}".format(servo_id = servo_id, angle = angle))
		move_group = self.move_group
		if servo_id is None:
			if not isinstance(angle, list):
				joint_goal = angle.tolist()[:7]
			else:
				joint_goal = angle[:7]
			assert len(joint_goal) == 7 or (len(joint_goal) == 9)
			# The go command can be called with joint values, poses, or without any
			# parameters if you have already set the pose or joint target for the group
			move_group.go(joint_goal[:7], wait=True)

			# Calling ``stop()`` ensures that there is no residual movement
			move_group.stop()
		else:
			assert isinstance(angle, int) or isinstance(angle, float)
			move_group[servo_id] = angle
			# The go command can be called with joint values, poses, or without any
			# parameters if you have already set the pose or joint target for the group
			move_group.go(joint_goal, wait=True)

			# Calling ``stop()`` ensures that there is no residual movement
			move_group.stop()
			# For testing:
		current_joints = move_group.get_current_joint_values()
		return all_close(joint_goal, current_joints, 0.01)

	def get_servo_angle(self, servo_id=None, is_radian=True):

		if self.dry_run:
			return 0, self.rest_pose
		if servo_id is not None:
			assert servo_id >= 1 and servo_id <= 7
		if servo_id is None:
			return 0, self.move_group.get_current_joint_values()

		else:
			return 0, self.move_group.get_current_joint_values()[servo_id]

	def get_joint_states(self):
		if self.dry_run: 
			return 0, self.rest_pose
		move_group = self.move_group
		return move_group.get_current_joint_values()

	def go_to_pose_goal(self, x=0.3067, y=-0.00035, z=0.5899):
		# Copy class variables to local variables to make the web tutorials more clear.
		# In practice, you should use the class variables directly unless you have a good
		# reason not to.
		if self.dry_run:
			return True
		move_group = self.move_group

		# BEGIN_SUB_TUTORIAL plan_to_pose
		##
		# Planning to a Pose Goal
		# ^^^^^^^^^^^^^^^^^^^^^^^
		# We can plan a motion for this group to a desired pose for the
		# end-effector:
		pose_goal = geometry_msgs.msg.Pose()

		current_pose = self.move_group.get_current_pose().pose
		# print(f"current pose is {current_pose}")
		pose_goal.orientation.x = 1
		pose_goal.orientation.y = 0
		pose_goal.orientation.z = 0
		pose_goal.orientation.w = 0
		pose_goal.position.x = x
		pose_goal.position.y = y  # + 0.098345534 # + 0.10 ## 0.098345534 is the offset: the dist between joint 8 and true end effector
		pose_goal.position.z = z
		move_group.set_pose_target(pose_goal)

		success = move_group.go(wait=True)
		move_group.stop()
		move_group.clear_pose_targets()

		current_pose = self.move_group.get_current_pose().pose
		return all_close(pose_goal, current_pose, 0.01)

	def approach_goal(self, x=0.3067, y=-0.00035, z=0.5899):
		# Copy class variables to local variables to make the web tutorials more clear.
		# In practice, you should use the class variables directly unless you have a good
		# reason not to.
		if self.dry_run:
			return True
		move_group = self.move_group

		# BEGIN_SUB_TUTORIAL plan_to_pose
		##
		# Planning to a Pose Goal
		# ^^^^^^^^^^^^^^^^^^^^^^^
		# We can plan a motion for this group to a desired pose for the
		# end-effector:
		pose_goal = geometry_msgs.msg.Pose()

		current_pose = self.move_group.get_current_pose().pose
		# print(f"current pose is {current_pose}")
		pose_goal.orientation.x = 1
		pose_goal.orientation.y = 0
		pose_goal.orientation.z = 0
		pose_goal.orientation.w = 0
		pose_goal.position.x = x
		pose_goal.position.y = y
		pose_goal.position.z = (
			z + 0.098345534
		)  # is the offset: the dist between joint 8 and true end effector
		move_group.set_pose_target(pose_goal)

		success = move_group.go(wait=True)
		move_group.stop()
		move_group.clear_pose_targets()

		current_pose = self.move_group.get_current_pose().pose
		pose_goal.orientation.x = 1
		pose_goal.orientation.y = 0
		pose_goal.orientation.z = 0
		pose_goal.orientation.w = 0
		pose_goal.position.x = x
		pose_goal.position.y = y
		pose_goal.position.z = z
		move_group.set_pose_target(pose_goal)

		success = move_group.go(wait=True)
		move_group.stop()
		move_group.clear_pose_targets()

		return all_close(pose_goal, current_pose, 0.01)

	def get_current_end_effector_position(self):
		if self.dry_run:
			return [-999.99,-999.99,-999.99]
		cur_pose = self.move_group.get_current_pose().pose
		return [cur_pose.position.x, cur_pose.position.y, cur_pose.position.z]

	def get_current_end_effector_orientation(self):
		if self.dry_run:
			return [0,0,0,1]
		cur_pose = self.move_group.get_current_pose().pose
		return [
			cur_pose.orientation.x,
			cur_pose.orientation.y,
			cur_pose.orientation.z,
			cur_pose.orientation.w,
		]


if __name__ == "__main__":
	import numpy as np

	demo = MoveGroupPythonInterface()
	demo.go_to_pose_goal()
