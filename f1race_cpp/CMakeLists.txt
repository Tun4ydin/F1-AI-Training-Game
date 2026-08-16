cmake_minimum_required(VERSION 3.20)
project(FormulaAILabNative VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)
find_package(nlohmann_json 3.10 REQUIRED)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui)
add_executable(formula_ai_lab_cpp
    src/main.cpp
    src/safe_algorithm.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer2.cpp
    ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp)
target_include_directories(formula_ai_lab_cpp PRIVATE
    ${IMGUI_DIR} ${IMGUI_DIR}/backends ${IMGUI_DIR}/misc/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)
target_link_libraries(formula_ai_lab_cpp PRIVATE
    PkgConfig::SDL2 nlohmann_json::nlohmann_json)
target_compile_definitions(formula_ai_lab_cpp PRIVATE
    F1RACE_CPP_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    F1RACE_PYTHON_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../f1race/saved_data"
    F1RACE_CAR_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../f1race/assets/cars")
if(MSVC)
    target_compile_options(formula_ai_lab_cpp PRIVATE /W4 /permissive-)
else()
    target_compile_options(formula_ai_lab_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME native_smoke COMMAND formula_ai_lab_cpp --smoke-test)
add_test(NAME native_simulation COMMAND formula_ai_lab_cpp --simulation-test)
add_test(NAME native_imgui_smoke COMMAND formula_ai_lab_cpp --ui-smoke)
set_tests_properties(native_imgui_smoke PROPERTIES
    ENVIRONMENT "SDL_VIDEODRIVER=dummy")
add_test(NAME native_imgui_small_window COMMAND formula_ai_lab_cpp --ui-smoke-small)
set_tests_properties(native_imgui_small_window PROPERTIES
    ENVIRONMENT "SDL_VIDEODRIVER=dummy")
add_test(NAME native_training_setup COMMAND formula_ai_lab_cpp --ui-training-setup)
set_tests_properties(native_training_setup PROPERTIES
    ENVIRONMENT "SDL_VIDEODRIVER=dummy")
add_test(NAME native_race_setup COMMAND formula_ai_lab_cpp --ui-race-setup)
set_tests_properties(native_race_setup PROPERTIES
    ENVIRONMENT "SDL_VIDEODRIVER=dummy")
add_test(NAME native_hotlap_setup COMMAND formula_ai_lab_cpp --ui-hotlap-setup)
set_tests_properties(native_hotlap_setup PROPERTIES
    ENVIRONMENT "SDL_VIDEODRIVER=dummy")
add_test(NAME native_track_studio COMMAND formula_ai_lab_cpp --ui-track-studio)
set_tests_properties(native_track_studio PROPERTIES
    ENVIRONMENT "SDL_VIDEODRIVER=dummy")
