project(OperaHPCPolydisperseBubble)

# Copy of the `bubble` case for polydisperse bubbles with one pressure per
# bubble (see POLYDISPERSE.md). Target and test names differ from `bubble`
# because CMake target names must be unique in the whole project.

file(COPY verification/mmm_vs_analytical.py DESTINATION ${PROJECT_BINARY_DIR})

# behaviour law (MFront -> src/libBehaviour.so in this build folder)
add_custom_command(
  OUTPUT "src/libBehaviour.so"
  COMMAND mfront --obuild --interface=generic ${PROJECT_BINARY_DIR}/Elasticity.mfront
  DEPENDS ${PROJECT_BINARY_DIR}/Elasticity.mfront
  COMMENT "Performing mfront code generation")

add_custom_target(lib-elasticity-poly ALL
  DEPENDS "src/libBehaviour.so")

# build executable

# only the sources of src/ (tools/ contains other programs with their own main)
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

# Tool to split a big Gmsh 2.2 mesh into MFEM parallel mesh files, to be read
# with `test-polybubble -pm 1` (see POLYDISPERSE.md). Only needed when the
# full mesh does not fit in memory on every process: uncomment the two lines
# below and add partition-mesh to the install line.
# add_executable(partition-mesh tools/PartitionMesh.cxx)
# target_link_libraries(partition-mesh PUBLIC mfem-mgis::MFEMMGIS)

# copy and install

file(COPY Elasticity.mfront mesh DESTINATION ${PROJECT_BINARY_DIR})

add_test(NAME PolyBubbleTest COMMAND test-polybubble --mesh mesh/mesh_sphere.msh --bubble-file mesh/single_bubble_ci.txt)

install(TARGETS test-polybubble DESTINATION bin)
