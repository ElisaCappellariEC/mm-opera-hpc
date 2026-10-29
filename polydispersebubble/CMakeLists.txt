project(OperaHPCPolydisperseBubble)

# Copy of the `bubble` case for polydisperse bubbles with one pressure per bubble.
# Code modified by
# \author E. Cappellari (POLIMI-CEA, 2026)

file(COPY verification/mmm_vs_analytical.py DESTINATION ${PROJECT_BINARY_DIR})

# scripts/ are run from the build folder.
# configure_file copies them again at each build when they change.
file(GLOB SCRIPTS CONFIGURE_DEPENDS scripts/*.py scripts/*.sh)
foreach(script ${SCRIPTS})
  get_filename_component(name ${script} NAME)
  configure_file(${script} ${PROJECT_BINARY_DIR}/${name} COPYONLY)
endforeach()

# behaviour law (MFront -> src/libBehaviour.so in this build folder)
add_custom_command(
  OUTPUT "src/libBehaviour.so"
  COMMAND mfront --obuild --interface=generic ${PROJECT_BINARY_DIR}/Elasticity.mfront
  DEPENDS ${PROJECT_BINARY_DIR}/Elasticity.mfront
  COMMENT "Performing mfront code generation")

add_custom_target(lib-elasticity-poly ALL
  DEPENDS "src/libBehaviour.so")

# build executable

# sources of src/
file( GLOB_RECURSE SRCS src/*.cxx )

add_executable(test-polybubble
  ${SRCS}
)

add_dependencies(test-polybubble lib-elasticity-poly)

target_link_libraries(test-polybubble
 PRIVATE tfel::TFELMath tfel::TFELException
 PUBLIC mfem-mgis::MFEMMGIS)

target_include_directories(test-polybubble
   PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Tool to split a whole Gmsh 2.2 mesh into one MFEM file per MPI process (-pm 1)
add_executable(partition-mesh tools/PartitionMesh.cxx)
target_link_libraries(partition-mesh PUBLIC mfem-mgis::MFEMMGIS)

# copy and install

file(COPY Elasticity.mfront mesh DESTINATION ${PROJECT_BINARY_DIR})

add_test(NAME PolyBubbleTest COMMAND test-polybubble --mesh mesh/mesh_sphere.msh --bubble-file mesh/single_bubble_ci.txt)

install(TARGETS test-polybubble partition-mesh DESTINATION bin)
