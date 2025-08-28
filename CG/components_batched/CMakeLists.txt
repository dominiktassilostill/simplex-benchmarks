CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

# find deal.II
MESSAGE("Find deal.II")
FIND_PACKAGE(deal.II 9.5 REQUIRED
        HINTS  ${DEAL_II_DIR} 
        )

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT("simplex benchmark")

# find likwid
MESSAGE("Find Likwid")
FIND_LIBRARY(LIKWID likwid HINTS ${LIKWID_DIR}/lib)
MESSAGE(STATUS "Using LIKWID found at ${LIKWID}")

DEAL_II_INITIALIZE_CACHED_VARIABLES()

SET(TARGET "application")
SET(TARGET_SRC
    ${TARGET}.cc)

ADD_EXECUTABLE(bench ${TARGET_SRC})

IF(EXISTS ${LIKWID})
  # The macro LIKWID_PERFMON is used inside Likwid, i.e., we need to pass
  # that precise macro into the Likwid machinery for compilation.
  TARGET_COMPILE_DEFINITIONS(bench PUBLIC LIKWID_PERFMON)
  TARGET_INCLUDE_DIRECTORIES(bench PUBLIC ${LIKWID_DIR}/include)
  TARGET_LINK_LIBRARIES(bench ${LIKWID})
ENDIF()

DEAL_II_SETUP_TARGET(bench)
