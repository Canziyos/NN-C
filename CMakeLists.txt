cmake_minimum_required(VERSION 3.16)

project(nnc VERSION 0.3.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

function(nnc_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(
            ${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        )
    endif()
endfunction()

function(nnc_generated_target target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    target_link_libraries(${target} PRIVATE nnc)
    nnc_warnings(${target})
endfunction()

add_library(nnc src/nn.c)
target_include_directories(nnc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
nnc_warnings(nnc)
if(NOT MSVC)
    target_link_libraries(nnc PUBLIC m)
endif()

add_executable(nnc_basic examples/basic.c)
target_link_libraries(nnc_basic PRIVATE nnc)

nnc_generated_target(nnc_generated examples/generated.c)
nnc_generated_target(nnc_trained_xor examples/trained_xor.c)

include(CTest)

if(BUILD_TESTING)
    add_executable(nnc_tests tests/test_forward.c)
    target_link_libraries(nnc_tests PRIVATE nnc)
    nnc_warnings(nnc_tests)

    nnc_generated_target(nnc_generated_tests tests/test_generated_model.c)
    nnc_generated_target(nnc_trained_xor_tests tests/test_trained_xor.c)

    add_test(NAME nnc_tests COMMAND nnc_tests)
    add_test(NAME nnc_generated_tests COMMAND nnc_generated_tests)
    add_test(NAME nnc_trained_xor_tests COMMAND nnc_trained_xor_tests)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME nnc_exporter_tests
            COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p test_export_model.py
        )
        add_test(
            NAME nnc_example_header_fresh
            COMMAND ${Python3_EXECUTABLE} tools/export_model.py
                models/example_model.json -o examples/generated_model.h
                --symbol example_model --check
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        add_test(
            NAME nnc_trained_model_fresh
            COMMAND ${Python3_EXECUTABLE} tools/train_xor.py
                --output models/trained_xor.json --check
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        add_test(
            NAME nnc_trained_header_fresh
            COMMAND ${Python3_EXECUTABLE} tools/export_model.py
                models/trained_xor.json -o examples/trained_xor_model.h
                --symbol trained_xor --check
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endif()
