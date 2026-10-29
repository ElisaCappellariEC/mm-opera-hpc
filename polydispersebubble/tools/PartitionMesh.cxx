/*!
 * \file   tools/PartitionMesh.cxx
 * \brief  Split a whole Gmsh 2.2 mesh into one MFEM parallel mesh file per
 *         MPI process, to be read by `test-polybubble -pm 1`.
 * \author E. Cappellari (POLIMI-CEA, 2026)
 *
 * With `-pm 1` (MeshReadMode = "Restart") each process reads only its own part, 
 * from the file `<prefix>` + rank on 6 digits (mfem::MakeParFilename) 
 *
 * This tool runs on ONE process: it reads the mesh once, the same way
 * mfem-mgis does, splits it with METIS (the same partitioning as
 * mfem::ParMesh in the "FromScratch" mode) and writes the parts one by one
 * (mfem::MeshPartitioner), with the nodes of the order 2 periodic mesh.
 *
 * Usage (N = number of processes of the later test-polybubble run):
 *   ./partition-mesh -m mesh.msh -np N -o mesh_par.
 *   mpirun -n N ./test-polybubble -pm 1 -m mesh_par. ...
 *
 * NB: the Gmsh files split by Gmsh (`gmsh -part N`) are not in this format
 * and can't be read by `-pm 1`: give this tool the whole mesh.
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "mfem.hpp"

int main(int argc, char** argv) {
  mfem::Mpi::Init(argc, argv);
  if (mfem::Mpi::WorldSize() != 1) {
    if (mfem::Mpi::Root()) {
      std::cerr << "partition-mesh runs on one process: launch it without "
                   "mpirun (or with -n 1)\n";
    }
    return EXIT_FAILURE;
  }

  const char* mesh_file = nullptr;
  const char* output_prefix = "mesh_par.";
  int number_of_parts = 0;
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Whole mesh (Gmsh 2.2, the one given to test-polybubble -m).",
                 true);
  args.AddOption(&number_of_parts, "-np", "--number-of-parts",
                 "Number of parts = number of MPI processes of the "
                 "test-polybubble run.",
                 true);
  args.AddOption(&output_prefix, "-o", "--output-prefix",
                 "Prefix of the output files: <prefix>000000, "
                 "<prefix>000001, ... Give the same prefix to "
                 "test-polybubble -pm 1 -m <prefix>.");
  args.Parse();
  if (!args.Good() || number_of_parts < 1) {
    args.PrintUsage(std::cout);
    return EXIT_FAILURE;
  }
  args.PrintOptions(std::cout);

  // Same arguments as mfem-mgis' mesh loading in the "FromScratch" mode:
  // generate_edges = 0, refine = 1, fix_orientation = true
  mfem::Mesh mesh(mesh_file, 0, 1, true);
  const auto* const nodes = mesh.GetNodes();
  std::cout << "mesh: " << mesh.GetNV() << " vertices, " << mesh.GetNE()
            << " elements, " << mesh.GetNBE() << " boundary elements, "
            << "geometry of order "
            << (nodes != nullptr ? nodes->FESpace()->GetMaxElementOrder() : 1)
            << std::endl;

  // METIS with the default method (1), as mfem::ParMesh(comm, mesh)
  mfem::MeshPartitioner partitioner(mesh, number_of_parts);
  std::cout << "partitioning done, writing " << number_of_parts << " files"
            << std::endl;

  mfem::MeshPart part;
  for (int i = 0; i != number_of_parts; ++i) {
    partitioner.ExtractPart(i, part);
    // ExtractPart leaves out the nodes: they carry the order 2 geometry and
    // the periodicity, so they are copied to the part too
    if (nodes != nullptr) {
      part.nodal_fes = partitioner.ExtractFESpace(part, *(nodes->FESpace()));
      part.nodes =
          partitioner.ExtractGridFunction(part, *nodes, *(part.nodal_fes));
    }
    const auto file_name = mfem::MakeParFilename(output_prefix, i);
    std::ofstream output(file_name);
    output.precision(16);
    part.Print(output);
    output.close();
    if (!output) {
      std::cerr << "can't write '" << file_name << "'\n";
      return EXIT_FAILURE;
    }
    if ((i + 1) % (number_of_parts >= 10 ? number_of_parts / 10 : 1) == 0) {
      std::cout << "  " << i + 1 << "/" << number_of_parts << " written"
                << std::endl;
    }
  }
  std::cout << "wrote " << mfem::MakeParFilename(output_prefix, 0) << " ... "
            << mfem::MakeParFilename(output_prefix, number_of_parts - 1)
            << std::endl;
  return EXIT_SUCCESS;
}
