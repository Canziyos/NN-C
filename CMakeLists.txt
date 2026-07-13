cmake_minimum_required(VERSION 3.16)

project(nnc VERSION 0.2.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(nnc src/nn.c)
target_include_directories(nnc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(nnc PRIVATE /W4)
else()
    target_compile_options(nnc PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    target_link_libraries(nnc PUBLIC m)
endif()

add_executable(nnc_basic examples/basic.c)
target_link_libraries(nnc_basic PRIVATE nnc)

include(CTest)

if(BUILD_TESTING)
    add_executable(nnc_tests tests/test_forward.c)
    target_link_libraries(nnc_tests PRIVATE nnc)

    if(MSVC)
        target_compile_options(nnc_tests PRIVATE /W4)
    else()
        target_compile_options(nnc_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    add_test(NAME nnc_tests COMMAND nnc_tests)
endif()
