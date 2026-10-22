/*!
 * \file   tools/PartitionMesh.cxx
 * \brief  Split a (big) Gmsh 2.2 mesh into one MFEM parallel mesh file per
 *         MPI process, to be read by `test-bubble -pm 1`.
 *
 * Why: with `-pm 0` (MeshReadMode = "FromScratch") every MPI process of
 * `test-bubble` reads the whole serial mesh before it is partitioned, so the
 * memory needed per process is the size of the full mesh. With `-pm 1`
 * (MeshReadMode = "Restart") each process only reads its own part, in the
 * MFEM parallel format, from the file `<prefix>` + rank on 6 digits
 * (mfem::MakeParFilename), e.g. mesh_par.000000, mesh_par.000001, ...
 *
 * This tool does the "FromScratch" reading once, the same way mfem-mgis does
 * it (serial mfem::Mesh, then mfem::ParMesh with the default METIS
 * partitioning), and saves each part with mfem::ParMesh::ParPrint.
 *
 * Usage (N = number of processes of the later test-bubble run):
 *   mpirun -n N ./partition-mesh -m mesh.msh -o mesh_par.
 *   mpirun -n N ./test-bubble -pm 1 -m mesh_par. ...
 *
 * NB: this step still needs the full mesh in memory on every process, but
 * only once: spread the processes over several nodes (few per node).
 * NB: Gmsh 4.1 files and meshes partitioned by Gmsh are not supported by
 * MFEM: convert to a single Gmsh 2.2 file first (see POLYDISPERSE.md).
 */

#include <fstream>
#include <iostream>
#include <string>

#include "mfem.hpp"

int main(int argc, char** argv) {
  mfem::Mpi::Init(argc, argv);
  const auto rank = mfem::Mpi::WorldRank();

  const char* mesh_file = nullptr;
  const char* output_prefix = "mesh_par.";
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Input mesh (Gmsh 2.2, the one given to test-bubble -m).",
                 true);
  args.AddOption(&output_prefix, "-o", "--output-prefix",
                 "Prefix of the output files: <prefix>000000, "
                 "<prefix>000001, ... Give the same prefix to "
                 "test-bubble -pm 1 -m <prefix>.");
  args.Parse();
  if (!args.Good()) {
    if (rank == 0) {
      args.PrintUsage(std::cout);
    }
    return EXIT_FAILURE;
  }
  if (rank == 0) {
    args.PrintOptions(std::cout);
  }

  // Same arguments as mfem-mgis' sequential mesh loading:
  // generate_edges = 0, refine = 1, fix_orientation = true
  mfem::Mesh serial_mesh(mesh_file, 0, 1, true);
  if (rank == 0) {
    std::cout << "serial mesh: " << serial_mesh.GetNE() << " elements, "
              << serial_mesh.GetNBE() << " boundary elements\n";
  }

  // Partition the mesh between the MPI processes (METIS, as in mfem-mgis)
  mfem::ParMesh parallel_mesh(MPI_COMM_WORLD, serial_mesh);
  // the serial mesh is no longer needed: free its memory before writing
  serial_mesh.Clear();

  // Each process writes its part
  const auto file_name = mfem::MakeParFilename(output_prefix, rank);
  std::ofstream output(file_name);
  if (!output) {
    std::cerr << "rank " << rank << ": can't open '" << file_name << "'\n";
    return EXIT_FAILURE;
  }
  output.precision(16);
  parallel_mesh.ParPrint(output);
  if (rank == 0) {
    std::cout << "wrote " << mfem::Mpi::WorldSize() << " files: "
              << mfem::MakeParFilename(output_prefix, 0) << " ... "
              << mfem::MakeParFilename(output_prefix,
                                       mfem::Mpi::WorldSize() - 1)
              << "\n";
  }
  return EXIT_SUCCESS;
}
