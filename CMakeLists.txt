cmake_minimum_required(VERSION 3.1)
set(project_name "stereo")
project(${project_name})

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wconversion -Wdisabled-optimization -Wendif-labels -Wfloat-equal -Winit-self -Winline -Wlogical-op -Wmissing-include-dirs -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Wpacked -Wpointer-arith -Wredundant-decls -Wsign-promo -Wswitch-default -Wswitch-enum -Wunsafe-loop-optimizations -Wvariadic-macros -Wwrite-strings")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s -DNDEBUG -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g3 -Og -pg")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -s -DNDEBUG -march=native -mtune=native")
# CMAKE_BUILD_TYPE settings.

find_package(OpenCV REQUIRED)

add_executable(${project_name}
  ./src/main.cpp
)

target_include_directories(${project_name} PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${project_name}
  ${OpenCV_LIBS}
)
