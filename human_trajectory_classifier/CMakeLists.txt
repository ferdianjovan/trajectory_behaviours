cmake_minimum_required(VERSION 2.8.3)
project(human_trajectory_classifier)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  human_trajectory
  rospy
  std_msgs
  message_generation
)

# find_package(Boost REQUIRED COMPONENTS system)
catkin_python_setup()

###################################
## catkin specific configuration ##
###################################

add_message_files(
    DIRECTORY msg
    FILES HumanTrajectoryClassification.msg TrajectoryType.msg
)

add_action_files(
    DIRECTORY action
    FILES TrajectoryClassification.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib
  actionlib_msgs
)

catkin_package(
  CATKIN_DEPENDS human_trajectory geometry_msgs rospy std_msgs actionlib actionlib_msgs message_runtime
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
  scripts/client_test.py
  scripts/trajectory_classifier.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_human_trajectory_classifier.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
