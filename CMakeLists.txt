cmake_minimum_required(VERSION 3.16)

project(nnc VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(nnc
    Model.c
    init.c
    math_utils.c
)

target_include_directories(nnc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(NOT MSVC)
    target_link_libraries(nnc PUBLIC m)
endif()

add_executable(nn_example
    main.c
    debug_utils.c
)

target_link_libraries(nn_example PRIVATE nnc)

include(CTest)

if(BUILD_TESTING)
    add_executable(test_forward
        tests/test_forward.c
    )

    target_link_libraries(test_forward PRIVATE nnc)

    add_test(
        NAME deterministic_forward
        COMMAND test_forward
    )
endif()
