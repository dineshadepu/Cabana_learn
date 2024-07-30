add_executable(01ParticleArrayCreation particle_array_creation.cpp)
target_link_libraries(01ParticleArrayCreation LINK_PUBLIC Cabana::Core)

add_executable(TwoBlocksColliding two_block_colliding.cpp)
target_link_libraries(TwoBlocksColliding LINK_PUBLIC Cabana::Core)

install(TARGETS TwoBlocksColliding DESTINATION ${CMAKE_INSTALL_BINDIR})



add_definitions(-w)
