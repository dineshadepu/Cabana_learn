set(HEADERS
  CabanaLearn_Particles.hpp
  )

set(SOURCES
  CabanaLearn_Particles.cpp

  )

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(cabanalearn ${SOURCES})

target_link_libraries(cabanalearn
  Cabana::Core
  Cabana::Grid
  )

target_include_directories(cabanalearn
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

install(TARGETS cabanalearn LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
