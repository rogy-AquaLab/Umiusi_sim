cmake_minimum_required(VERSION 3.8)
project(umiusi_sim NONE)

# Metapackage: aggregates the UMIUSI simulation / RL sub-packages.
# It carries no build targets of its own; the sub-packages listed as
# exec_depend in package.xml are built independently by colcon.

find_package(ament_cmake REQUIRED)

ament_package()
