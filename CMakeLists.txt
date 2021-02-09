cmake_minimum_required(VERSION 3.0)
project( grid_feature)

# Output binary file directory setting
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# source files
set ( SRC_FILES test/demo.cc src/feature.cc)

# OpenCV 
find_package( OpenCV REQUIRED )

include_directories( 
${PROJECT_SOURCE_DIR}
)
message(STATUS ${OpenCV})


include_directories( include )

add_executable( ${PROJECT_NAME} ${SRC_FILES} )


target_link_libraries(${PROJECT_NAME} 
${OpenCV_LIBS}
)
