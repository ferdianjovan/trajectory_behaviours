cmake_minimum_required(VERSION 2.8.3)
project(occurrence_learning)

find_package(catkin REQUIRED COMPONENTS
  tf
  rospy
  actionlib
  actionlib_msgs
  mongodb_store
  std_msgs
  message_generation
  soma_geospatial_store
  strands_navigation_msgs
  soma_msgs
)

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  FILES
  OccurrenceRate.msg
  RegionObservationTime.msg
)

add_service_files(
  FILES
  TrajectoryOccurrenceRate.srv
)

add_action_files(
  FILES
  OccurrenceRateLearning.action
)

generate_messages(
  DEPENDENCIES
  std_msgs actionlib actionlib_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES poisson_stuff
  CATKIN_DEPENDS tf rospy std_msgs message_runtime soma_geospatial_store actionlib actionlib_msgs mongodb_store strands_navigation_msgs soma_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

install(PROGRAMS
  scripts/tof_plot.py
  scripts/tof_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
