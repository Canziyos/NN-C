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

add_executable(nnc_generated examples/generated.c)
target_include_directories(nnc_generated PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_link_libraries(nnc_generated PRIVATE nnc)

include(CTest)

if(BUILD_TESTING)
    add_executable(nnc_tests tests/test_forward.c)
    target_link_libraries(nnc_tests PRIVATE nnc)

    add_executable(nnc_generated_tests tests/test_generated_model.c)
    target_include_directories(
        nnc_generated_tests
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples
    )
    target_link_libraries(nnc_generated_tests PRIVATE nnc)

    if(MSVC)
        target_compile_options(nnc_tests PRIVATE /W4)
        target_compile_options(nnc_generated_tests PRIVATE /W4)
    else()
        target_compile_options(nnc_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        target_compile_options(
            nnc_generated_tests
            PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        )
    endif()

    add_test(NAME nnc_tests COMMAND nnc_tests)
    add_test(NAME nnc_generated_tests COMMAND nnc_generated_tests)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME nnc_exporter_tests
            COMMAND
                ${Python3_EXECUTABLE}
                -m unittest
                discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/tests
                -p test_export_model.py
        )
        add_test(
            NAME nnc_generated_header_fresh
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/export_model.py
                ${CMAKE_CURRENT_SOURCE_DIR}/models/example_model.json
                --output
                ${CMAKE_CURRENT_SOURCE_DIR}/examples/generated_model.h
                --symbol
                example_model
                --check
        )
    endif()
endif()
