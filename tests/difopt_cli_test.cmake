if(NOT DEFINED DIFC OR NOT DEFINED DIFOPT OR NOT DEFINED TEST_DIRECTORY)
  message(FATAL_ERROR "difopt CLI test is missing executable or directory")
endif()

file(REMOVE_RECURSE "${TEST_DIRECTORY}")
file(MAKE_DIRECTORY "${TEST_DIRECTORY}")

execute_process(
  COMMAND "${DIFC}" make-h3-block-raw-bf16
          "${TEST_DIRECTORY}/base.difir" 2 4 1 4 8 2 64 streamed
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "cannot create difopt fixture: ${output}${error}")
endif()

execute_process(
  COMMAND "${DIFOPT}" weight-placement
          --program "${TEST_DIRECTORY}/base.difir"
          --output "${TEST_DIRECTORY}/optimized.difir"
          --device-budget-mib 4 --evaluations 3
          --plan-output "${TEST_DIRECTORY}/winner.difplan"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT output MATCHES "OPTIMIZE PASS")
  message(FATAL_ERROR "weight placement CLI failed: ${output}${error}")
endif()

execute_process(
  COMMAND "${DIFOPT}" replay
          --program "${TEST_DIRECTORY}/base.difir"
          --plan "${TEST_DIRECTORY}/winner.difplan"
          --output "${TEST_DIRECTORY}/replayed.difir"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT output MATCHES "REPLAY PASS")
  message(FATAL_ERROR "optimization replay CLI failed: ${output}${error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${TEST_DIRECTORY}/optimized.difir"
          "${TEST_DIRECTORY}/replayed.difir"
  RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "optimization plan replay is not byte-identical")
endif()

execute_process(
  COMMAND "${DIFOPT}" replay
          --program "${TEST_DIRECTORY}/base.difir"
          --plan "${TEST_DIRECTORY}/winner.difplan"
          --output "${TEST_DIRECTORY}/replayed.difir"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT error MATCHES "refusing to overwrite")
  message(FATAL_ERROR "optimization replay did not refuse overwrite")
endif()
