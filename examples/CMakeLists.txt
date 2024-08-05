add_executable(01ParticleArrayCreation particle_array_creation.cpp)
target_link_libraries(01ParticleArrayCreation PRIVATE cabanalearn)

add_executable(TwoBlocksColliding two_block_colliding.cpp)
target_link_libraries(TwoBlocksColliding PRIVATE cabanalearn)

add_executable(Integrate integrate.cpp)
target_link_libraries(Integrate PRIVATE cabanalearn)


install(TARGETS Integrate 01ParticleArrayCreation TwoBlocksColliding DESTINATION ${CMAKE_INSTALL_BINDIR})



add_definitions(-w)
