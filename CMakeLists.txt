cmake_minimum_required(VERSION 3.8)
project(thesis_rain_sim)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Pure-Python helpers (rain_models, fog_dust, dust_geometry, weather) -> importable module
ament_python_install_package(${PROJECT_NAME})

# Python nodes -> installed as executables under lib/<pkg>
install(PROGRAMS
  scripts/weather_model_node.py
  scripts/clustering_denoiser_node.py
  scripts/ground_truth_logger.py
  scripts/run_missions.py
  scripts/analyse_results.py
  scripts/run_sweep.sh
  scripts/overnight.sh
  scripts/plot_rain_models.py
  DESTINATION lib/${PROJECT_NAME}
)

# Data directories
install(DIRECTORY
  launch
  worlds
  models
  config
  rviz
  behavior_trees
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_rain_models test/test_rain_models.py
    TIMEOUT 120)
  ament_add_pytest_test(test_fog_dust test/test_fog_dust.py
    TIMEOUT 120)
endif()

ament_package()
