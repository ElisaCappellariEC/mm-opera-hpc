/*!
 * \file   main.cxx
 * \brief  Main program for bubble fracture simulation in porous materials
 * \date   29/09/2024
 *
 * This program simulates the mechanical behavior of a material containing
 * bubbles under pressure. It identifies which bubbles are likely to break based
 * on the principal stress field around them.
 */

#include <cmath>
#include <limits>
#include <cstdlib>
#include <fstream>

#include "MGIS/Function/TFEL/Mechanics.hxx"

#include "MFEMMGIS/Config.hxx"
#include "MFEMMGIS/LinearSolverFactory.hxx"
#include "MFEMMGIS/L2Projection.hxx"
#include "MFEMMGIS/Material.hxx"
#include "MFEMMGIS/MechanicalPostProcessings.hxx"
#include "MFEMMGIS/PartialQuadratureFunction.hxx"
#include "MFEMMGIS/PartialQuadratureSpace.hxx"
#include "MFEMMGIS/PeriodicNonLinearEvolutionProblem.hxx"
#include "MFEMMGIS/Profiler.hxx"
#include "MFEMMGIS/UniformImposedPressureBoundaryCondition.hxx"

#include "Common/Memory.hxx"
#include "Common/Print.hxx"
#include "OperaHPC/BubbleDescription.hxx"
#include "OperaHPC/Utilities.hxx"

#include "mfem/general/forall.hpp"
#include "mfem/fem/datacollection.hpp"

namespace mfem_mgis {
  template <unsigned short N>
  static bool computeHydrostaticPressure_implementation(
      Context& ctx,
      PartialQuadratureFunction& hyp,
      const Material& m,
      const Material::StateSelection s) {
    using namespace mgis::function;
    const auto osig = getThermodynamicForce(ctx, m, "Stress", s);
    if (isInvalid(osig)) {
      return {};
    }
    const auto ok = *osig | as_stensor<N> | hydrostatic_stress | hyp;
    if (!ok) {
      return ctx.registerErrorMessage(
          "computeHydrostaticPressure: computation of the hydrostatic "
          "stress failed");
    }
    return true;
  }  // end of computeHydrostaticPressure_implementation

  static bool computeHydrostaticPressure(Context& ctx,
                                         PartialQuadratureFunction& hyp,
                                         const Material& m,
                                         const Material::StateSelection s) {
    if ((m.b.hypothesis == Hypothesis::PLANESTRESS) ||
        (m.b.hypothesis == Hypothesis::PLANESTRAIN)) {
      return computeHydrostaticPressure_implementation<2>(ctx, hyp, m, s);
    } else if (m.b.hypothesis == Hypothesis::TRIDIMENSIONAL) {
      return computeHydrostaticPressure_implementation<3>(ctx, hyp, m, s);
    }
    return ctx.registerErrorMessage(
        "computeHydrostaticPressure: unsupported modelling hypothesis");
  }  // end of computeHydrostaticPressure

}  // namespace mfem_mgis

namespace opera_hpc {

  /**
   * \brief Represents a bubble in the material with its state (broken or
   * intact)
   *
   * Extends BubbleDescription to add a boolean flag indicating whether
   * the bubble has broken during the simulation
   */
  struct Bubble : BubbleDescription {
    bool broken = false;  ///< True if the bubble has fractured
  };

  /**
   * \brief Check if all bubbles in the system have broken
   *
   * \param bubbles Vector of all bubbles to check
   * \return true if all bubbles are broken, false otherwise
   */
  bool areAllBroken(const std::vector<Bubble>& bubbles) {
    for (const auto& b : bubbles) {
      if (!b.broken) {
        return false;
      }
    }
    return true;
  }
}  // end of namespace opera_hpc

/*!
 * \brief Value used for "no stress found near this bubble" (no integration
 * point in its search sphere, or none above the threshold). Any real stress
 * is larger, which makes the max-reduction over MPI processes simple.
 */
constexpr mfem_mgis::real no_stress_value =
    -std::numeric_limits<mfem_mgis::real>::max();

/**
 * \brief Record storing bubble information for output
 *
 * Contains the boundary ID, location, and maximum stress value
 * found near a bubble during the analysis
 */
struct BubbleInfoRecord {
  mfem_mgis::size_type boundary_id;  ///< Boundary identifier of the bubble
  std::array<mfem_mgis::real, 3u>
      location;                  ///< 3D coordinates of max stress location
  mfem_mgis::real stress_value;  ///< Maximum principal stress value

  /**
   * \brief Constructor
   *
   * \param bid Boundary identifier
   * \param loc 3D location array
   * \param s_val Stress value at that location
   */
  BubbleInfoRecord(const mfem_mgis::size_type bid,
                   const std::array<mfem_mgis::real, 3u>& loc,
                   const mfem_mgis::real s_val)
      : boundary_id(bid), stress_value(s_val) {
    location[0] = loc[0];
    location[1] = loc[1];
    location[2] = loc[2];
  }
};

/**
 * \brief Execute post-processing operations on the problem
 *
 * \tparam Problem Type of the mechanical problem
 * \param p The problem instance
 * \param start Start time for post-processing
 * \param end End time for post-processing
 */
template <typename Problem>
bool post_process(mfem_mgis::Context& ctx,
                  Problem& p,
                  double start,
                  double end) {
  return p.executePostProcessings(ctx, start, end);
}

/**
 * \brief Physical parameters for bubble fracture simulation
 *
 * Contains the key physical parameters that control the mechanical
 * behavior of bubbles and the stress analysis criteria used to
 * determine bubble fracture.
 */
struct PhysicalParameters {
  /*!
   * Pressure multiplier: the pressure imposed in bubble i is
   * bubble_pressure * p_i, where p_i is read from the 6th column of the
   * bubble file (p_i = 1 when the file only has 5 columns, which gives back
   * the original "same pressure everywhere" behaviour).
   */
  mfem_mgis::real bubble_pressure = 1.0;
  mfem_mgis::real d_min =
      0.950;  ///< Distance threshold for associating stress with bubbles
  /*!
   * Thickness of the shell around each bubble in which the maximum first
   * principal stress is searched (same unit as the mesh, µm).
   * - delta > 0: the search radius of bubble i is radius_i + delta. This is
   *   needed for polydisperse populations: a fixed radius cannot be larger
   *   than the big bubbles and smaller than the gap around the small ones.
   * - delta <= 0 (default): the fixed radius d_min is used for every bubble
   *   (original behaviour, designed for a single bubble size).
   */
  mfem_mgis::real delta = 0;
};

/**
 * \brief Structure holding all simulation parameters
 *
 * Contains default values for mesh files, material properties,
 * numerical parameters, and output options
 */
struct TestParameters : PhysicalParameters {
  const char* mesh_file = "mesh/single_sphere.msh";  ///< Path to mesh file
  const char* behaviour = "Elasticity";              ///< Material behavior law
  const char* library = "src/libBehaviour.so";       ///< Material library path
  const char* bubble_file =
      "mesh/single_bubble.txt";                  ///< Bubble definitions file
  const char* testcase_name = "TestCaseBubble";  ///< Name for output files
  int parallel_mesh = 0;    ///< Flag for parallel mesh format
  int order = 2;            ///< Finite element order
  int refinement = 0;       ///< Mesh refinement level
  int post_processing = 1;  ///< Enable post-processing (1=yes)
  int verbosity_level = 0;  ///< Output verbosity level
  mfem_mgis::real scale_factor_vp =
      0.9;  ///< Scale factor for principal stress threshold
  /*!
   * Preconditioner of the linear solver (HyprePCG). The original code used
   * HypreDiagScale, which needs more than 1000 iterations on the default
   * mesh (mesh/single_sphere.msh): since MFEM 4.10 / mfem-mgis develop, the
   * solver then reports a failure ("the non linear solver did not converge").
   */
  const char* linear_solver_preconditioner = "HypreBoomerAMG";
  //! Maximum number of iterations of the linear solver (hypre default: 1000)
  int linear_solver_maximum_iterations = 5000;
};

/**
 * \brief Parse command-line arguments and fill parameter structure
 *
 * \param args Command-line argument parser
 * \param p Parameter structure to fill
 */
void fill_parameters(mfem::OptionsParser& args, TestParameters& p) {
  args.AddOption(&p.mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&p.parallel_mesh, "-pm", "--parallel-mesh",
                 "Parallel mesh format or not");
  args.AddOption(&p.library, "-l", "--library", "Material library.");
  args.AddOption(&p.bubble_file, "-f", "--bubble-file",
                 "File containing the bubbles.");
  args.AddOption(&p.order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&p.refinement, "-r", "--refinement",
                 "refinement level of the mesh, default = 0");
  args.AddOption(&p.post_processing, "-p", "--post-processing",
                 "export the stress fields for ParaView (1) or not (0). The "
                 "bubble CSV and sig_hydro.txt are always written.");
  args.AddOption(&p.verbosity_level, "-v", "--verbosity-level",
                 "choose the verbosity level");
  args.AddOption(&p.scale_factor_vp, "-sf", "--scale-factor-vp",
                 "Scaling factor for the principal stress: only the points "
                 "above sf * (maximum first principal stress) are used in "
                 "the per-bubble search. Use a value <= 0 to use all the "
                 "points, so that every bubble gets a value.");
  args.AddOption(&p.testcase_name, "-n", "--name-case",
                 "Name of the testcase.");
  args.AddOption(
      &p.d_min, "-dm", "--dmin",
      "Distance to evaluate bubble rupture, in µm, default = 0.950.");
  args.AddOption(&p.delta, "-dl", "--delta",
                 "If > 0, the stress of bubble i is searched within "
                 "radius_i + delta of its center (instead of dmin). "
                 "Recommended for polydisperse bubbles, default = 0.");
  args.AddOption(&p.bubble_pressure, "-pr", "--pref",
                 "Pressure multiplier: bubble i gets pref * p_i, p_i being "
                 "the 6th column of the bubble file (1 if absent), "
                 "default = 1");
  args.AddOption(&p.linear_solver_preconditioner, "-pc", "--preconditioner",
                 "Preconditioner of the linear solver: HypreBoomerAMG "
                 "(default), HypreDiagScale (original), HypreILU, ...");
  args.AddOption(&p.linear_solver_maximum_iterations, "-li",
                 "--linear-solver-iterations",
                 "Maximum number of iterations of the linear solver, "
                 "default = 5000");

  args.Parse();

  // Validate arguments
  if (!args.Good()) {
    if (mfem_mgis::isMPIMainProcessForDefaultCommunicator()) args.PrintUsage(std::cout);
    mfem_mgis::finalize();
    exit(0);
  }
  if (p.mesh_file == nullptr) {
    if (mfem_mgis::isMPIMainProcessForDefaultCommunicator())
      std::cout << "ERROR: Mesh file missing" << std::endl;
    args.PrintUsage(std::cout);
    mfem_mgis::abort(EXIT_FAILURE);
  }
  if (mfem_mgis::isMPIMainProcessForDefaultCommunicator()) args.PrintOptions(std::cout);
}

/**
 * \brief Write comma-separated values to a file stream
 *
 * Variadic template function to write any number of arguments
 * separated by commas, ending with a newline
 *
 * \param file_to_write Output file stream
 * \param args Variadic arguments to write
 */
void write_message(std::ofstream& file_to_write,
                   const auto& first,
                   const auto&... args) {
  // "a,b,c" without a trailing comma, so that the header has as many
  // columns as the data lines (the original version wrote "a,b,c,")
  file_to_write << first;
  ((file_to_write << "," << args), ...);
  file_to_write << std::endl;
}

/**
 * \brief Write bubble information records to a CSV file
 *
 * One line per bubble:
 *   bubble_id, radius, pressure, sigma_I_max, x, y, z
 * where pressure is the pressure read from the bubble file (before the
 * `--pref` multiplier), sigma_I_max the maximum first principal stress found
 * in the search sphere of the bubble and (x, y, z) where it was found.
 * Bubbles without any value are written with "nan".
 *
 * \param file_to_write Output file stream
 * \param bubbles Bubbles, in the same order as bubble_infos
 * \param bubble_infos Vector of bubble information records to write
 */
void write_bubble_infos(std::ofstream& file_to_write,
                        const std::vector<opera_hpc::Bubble>& bubbles,
                        const std::vector<BubbleInfoRecord>& bubble_infos) {
  file_to_write.precision(12);
  write_message(file_to_write, "bubble_id", "radius", "pressure",
                "sigma_I_max", "x", "y", "z");
  for (std::size_t i = 0; i != bubble_infos.size(); ++i) {
    const auto& b = bubbles[i];
    const auto& entry = bubble_infos[i];
    file_to_write << b.boundary_identifier << "," << b.radius << ","
                  << b.pressure << ",";
    if (entry.stress_value == no_stress_value) {
      file_to_write << "nan,nan,nan,nan\n";
      continue;
    }
    file_to_write << entry.stress_value;
    for (const auto& el : entry.location) {
      file_to_write << "," << el;
    }
    file_to_write << "\n";
  }
}

/**
 * \brief Calculate the average hydrostatic stress over a material domain
 *
 * Integrates the hydrostatic pressure field over all elements and
 * computes the volume-averaged value using MPI reduction if needed
 *
 * \tparam parallel Whether running in parallel mode
 * \param os Output stream for results
 * \param prob The nonlinear evolution problem
 * \param f Quadrature function containing the pressure field
 */
template <bool parallel>
static void calculateAverageHydrostaticStress(
    mfem_mgis::attributes::MayAbort,
    mfem_mgis::Context& ctx,
    std::ostream& os,
    const mfem_mgis::NonLinearEvolutionProblemImplementation<parallel>& prob,
    const mfem_mgis::ImmutablePartialQuadratureFunctionView& f) {
  auto or_die = ctx.getFatalFailureHandler();
  const auto& s = f.getPartialQuadratureSpace();
  const auto& fed = s.getFiniteElementDiscretization();
  const auto& fespace = fed.getFiniteElementSpace<parallel>();
  const auto l = s.getLocation();
  if(isInvalid(l.material_identifier)){
    mfem_mgis::abort("calculateAverageHydrostaticStress only works on materials");
  }
  const auto m = l.material_identifier->id;
  
  mfem_mgis::real integral = 0.;
  mfem_mgis::real volume = 0.;

  // Loop over all elements in the mesh
  for (mfem_mgis::size_type i = 0; i != fespace.GetNE(); ++i) {
    // Skip elements not in this material
    if (fespace.GetAttribute(i) != m) {
      continue;
    }
    const auto& fe = *(fespace.GetFE(i));
    auto& tr = *(fespace.GetElementTransformation(i));
    const auto& ir = s.getIntegrationRule(fe, tr);

    // Integrate over quadrature points
    for (mfem_mgis::size_type g = 0; g != ir.GetNPoints(); ++g) {
      const auto& ip = ir.IntPoint(g);
      tr.SetIntPoint(&ip);
      const auto& bi = prob.getBehaviourIntegrator(ctx, m, 0) | or_die;
      const auto w = bi.getIntegrationPointWeight(tr, ip);
      integral += f.getIntegrationPointValue(i, g) * w;
      volume += w;
    }
  }

  // MPI reduction to get global integral and volume
  mfem_mgis::real global_volume = 0.;
  mfem_mgis::real global_integral = 0.;
  MPI_Reduce(&integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // Write results on rank 0
  if (mfem_mgis::isMPIMainProcessForDefaultCommunicator()) {
    os << global_volume << "," << global_integral << ","
       << global_integral / global_volume << "\n";
  }
}  // end of calculateAverageHydrostaticStress

/**
 * \brief Main program
 *
 * Workflow:
 * 1. Initialize stuff and parse command-line arguments
 * 2. Load mesh and bubble definitions
 * 3. Set up the finite element problem
 * 4. Apply pressure boundary conditions on bubbles (pref * pressure of
 *    each bubble)
 * 5. Solve the mechanical equilibrium
 * 6. Keep the integration points above the stress threshold (all of them
 *    if sf <= 0), on each MPI process separately
 * 7. For each bubble, find the maximum first principal stress within its
 *    search sphere (radius + delta, or dmin), then combine the results of
 *    all the MPI processes on process 0
 * 8. Export results and post-process data
 */

int main(int argc, char** argv) {
  using namespace mfem_mgis::Profiler::Utils;

  // Initialize MPI and profiling
  mfem_mgis::initialize(argc, argv);
  auto ctx = mfem_mgis::Context();
  auto or_die = ctx.getFatalFailureHandler();
  ctx.enableProfiling(true);
  // Open output file for bubble stress results
  std::string bubbles_selected = "bubbles_and_stresses_selected.txt";
  std::string hydrostatic_stress_f = "sig_hydro.txt";
  std::ofstream outfile_sighydr(hydrostatic_stress_f);
  std::ofstream output_file_locations(bubbles_selected);

  if (!output_file_locations.is_open()) {
    std::cerr << "Failed to open the file: " << bubbles_selected << std::endl;
    return EXIT_FAILURE;
  }

  if (!outfile_sighydr.is_open()) {
    std::cerr << "Failed to open the file: " << hydrostatic_stress_f
              << std::endl;
    return EXIT_FAILURE;
  }

  // Parse command-line parameters
  TestParameters p;
  mfem::OptionsParser args(argc, argv);
  fill_parameters(args, p);

  print_memory_footprint("[Start]");

  // Physical parameters
  mfem_mgis::real dmin = p.d_min;

  // Load bubble descriptions from file
  auto bubbles = [p] {
    auto r = std::vector<opera_hpc::Bubble>{};
    for (const auto& d : opera_hpc::BubbleDescription::read(p.bubble_file)) {
      auto b = opera_hpc::Bubble{};
      static_cast<opera_hpc::BubbleDescription&>(b) = d;
      r.push_back(b);
    }
    return r;
  }();
  // Create finite element discretization
  print_memory_footprint("[Building problem ...]");
  auto fed =
      mfem_mgis::construct<mfem_mgis::FiniteElementDiscretization>(
          ctx,
          mfem_mgis::Parameters{
              {"MeshFileName", p.mesh_file},
              {"MeshReadMode", p.parallel_mesh ? "Restart" : "FromScratch"},
              {"FiniteElementFamily", "H1"},  // Continuous Lagrange elements
              {"FiniteElementOrder", p.order},
              {"UnknownsSize", 3},  // 3D displacement field
              {"NumberOfUniformRefinements", p.refinement},
              {"Parallel", true}}) |
      or_die;

  // Define the nonlinear evolution problem
  auto problem =
      construct<mfem_mgis::PeriodicNonLinearEvolutionProblem>(ctx, fed) |
      or_die;
  print_memory_footprint("[Building problem done]");

  print_mesh_information(problem.getImplementation<true>());

  // Set macroscopic strain to zero
  std::vector<mfem_mgis::real> e(6, mfem_mgis::real{0});
  problem.setMacroscopicGradientsEvolution(
      [e](const mfem_mgis::real) { return e; });
  // Apply pressure boundary conditions on each bubble: one boundary
  // condition per bubble, on the mesh boundary whose tag is the bubble
  // identifier (the "Physical Surface" of the bubble in the .geo file).
  // NB: the lambda captures a reference to the bubble, which is safe because
  // the vector `bubbles` is not resized after this point.
  for (const auto& b : bubbles) {
    problem.addBoundaryCondition(
        ctx, mfem_mgis::make_unique<
                 mfem_mgis::UniformImposedPressureBoundaryCondition>(
                 ctx, problem.getFiniteElementDiscretization(),
                 b.boundary_identifier,
                 [&b, pref = p.bubble_pressure](const mfem_mgis::real) {
                   // If bubble is broken, pressure = 0; otherwise pressure =
                   // pref * (pressure of this bubble, read from the file).
                   // NB: `broken` is never set to true in this program, it
                   // is a leftover of an iterative version.
                   return b.broken ? 0 : pref * b.pressure;
                 }) |
                 or_die) |
        or_die;
  }

  // Configure linear solver
  int verbosity = p.verbosity_level;
  int post_processing = p.post_processing;
  mfem_mgis::Parameters solverParameters;
  solverParameters.insert(mfem_mgis::may_throw,
                          mfem_mgis::Parameters{{"VerbosityLevel", verbosity}});
  // Preconditioner chosen with -pc (the original code always used
  // HypreDiagScale) and maximum number of iterations set with -li (the
  // original code kept the hypre default, 1000)
  auto preconditioner =
      mfem_mgis::Parameters{{"Name", p.linear_solver_preconditioner}};
  solverParameters.insert(
      mfem_mgis::may_throw,
      mfem_mgis::Parameters{
          {"Preconditioner", preconditioner},
          {"Tolerance", 1e-10},
          {"MaximumNumberOfIterations", p.linear_solver_maximum_iterations}});

  // Set up conjugate gradient solver
  problem.setLinearSolver(ctx, "HyprePCG", solverParameters) | or_die;
  problem.setSolverParameters(ctx, {{"VerbosityLevel", 1},
                                    {"RelativeTolerance", 1e-8},
                                    {"AbsoluteTolerance", 0.},
                                    {"MaximumNumberOfIterations", 1}}) |
      or_die;
  // Add elastic material behavior
  problem.addBehaviourIntegrator(ctx, "Mechanics", 1, p.library, p.behaviour) |
      or_die;
  auto& m = problem.getMaterial(ctx, 1, 0) | or_die;

  // Set material properties (elastic constants and temperature)
  // The sphere is not meshed, thus no material properties need to be
  // considered for it. We set the elastic constants for the material
  // which are representative for UO2.
  // NB we are considering an isotropic purely elastic behaviour,
  // imposing only the two elastic moduli. Please note that since
  // we are scaling everything to µm, we consider the Elastic
  // modulus in N/µm² to have a consistent displacement field.
  for (auto& s : {&m.s0, &m.s1}) {
    mgis::behaviour::setMaterialProperty(ctx, *s, "YoungModulus",
                                         150e-3) |
        or_die;  // Young's modulus
    mgis::behaviour::setMaterialProperty(ctx, *s, "PoissonRatio",
                                         0.3) |
        or_die;  // Poisson's ratio
    mgis::behaviour::setExternalStateVariable(ctx, *s, "Temperature",
                                              293.15) |
        or_die;  // Room temperature
  }
  // Configure post-processing output
  if (post_processing) {
    auto results = std::vector<mfem_mgis::Parameter>{"Stress"};
    problem.addPostProcessing(
        ctx, "ParaviewExportIntegrationPointResultsAtNodes",
        {{"OutputFileName", p.testcase_name}, {"Results", results}}) |
        or_die;
  }

  // Partial Quadrature functions for post-processing needs
  // eig for the first eigenstress
  auto eig = mfem_mgis::PartialQuadratureFunction{
      m.getPartialQuadratureSpace(), 1};
  // hyp for the hydrostatic stress
  auto hyp = mfem_mgis::PartialQuadratureFunction{
      m.getPartialQuadratureSpace(), 1};

  mfem_mgis::size_type nstep{1};
  // Solve the mechanical equilibrium problem
  problem.solve(ctx, 0, 1) | or_die;
  // Find the maximum principal stress in the domain
  const auto r = opera_hpc::findFirstPrincipalStressValueAndLocation(
      ctx, problem.getMaterial(ctx, 1, 0) | or_die);

  std::vector<BubbleInfoRecord> bubbles_information;

  // Define stress threshold
  // - sf > 0 (original behaviour): only the integration points whose first
  //   principal stress is above sf * (global maximum) are kept, so only the
  //   bubbles close to the most loaded points get a value.
  // - sf <= 0: every integration point is kept, so every bubble gets its own
  //   maximum (needed for population studies, e.g. the FGR of the README).
  const auto max_vp_scaled = (p.scale_factor_vp > 0)
                                 ? p.scale_factor_vp * r.value
                                 : no_stress_value;

  // Get the locations where the stress exceeds the threshold. Only the
  // points owned by this MPI process are returned (no communication): the
  // per-bubble maxima computed below are combined over all the processes
  // afterwards. This gives the same result as gathering all the points on
  // every process, but with much less memory on big meshes.
  auto all_locations_and_stresses_above_threshold =
      opera_hpc::getLocalPointsAndStressAboveStressThreshold(
          problem.getMaterial(ctx, 1, 0) | or_die, max_vp_scaled);
  // Resize the bubble information vector to match the number of bubbles,
  // initializing each entry with default values (0 identifier, {0,0,0}
  // location, "no value found" stress)
  bubbles_information.resize(bubbles.size(),
                             BubbleInfoRecord(0, {0, 0, 0}, no_stress_value));

  {
    CatchTimeSection(ctx, "BubbleLoop");
    // copied here so that the lambda below does not capture `p`
    const auto delta = p.delta;

    // Sort all  points by their X-coordinate for efficient spatial
    // searching
    std::sort(all_locations_and_stresses_above_threshold.begin(),
              all_locations_and_stresses_above_threshold.end(),
              [](const auto& a, const auto& b) {
                return a.location[0] < b.location[0];
              });

    // Pointers to avoid repeated vector access
    const auto* bubbles_data = bubbles.data();
    const auto* points_data = all_locations_and_stresses_above_threshold.data();
    auto* results_data = bubbles_information.data();

    const auto num_bubbles = bubbles.size();
    const auto num_points = all_locations_and_stresses_above_threshold.size();

    // Parallel loop over all bubbles
    // NB it can run OMP on CPU parallelization if hypre was compiled with this
    // support otherwise it degenerates to a simple for. it can do something
    // exotic with gpu but idk.
    // NB: each MPI process only sees its own points here, so the result for
    // a bubble is the maximum over the part of its neighbourhood owned by
    // this process. The global maximum is computed after this loop.
    mfem::forall(num_bubbles, [=] MFEM_HOST_DEVICE(int i) {
      // 1. Get current bubble to process
      const auto& b = bubbles_data[i];

      // Search radius of this bubble (distance from its center):
      // - delta > 0: radius + delta, i.e. a shell of thickness delta around
      //   the bubble surface, which adapts to the size of each bubble;
      // - otherwise: the fixed distance dmin (original behaviour).
      const auto rs = (delta > 0) ? b.radius + delta : dmin;

      // Initialize tracking variables for maximum stress point near this
      // bubble. Starting from "no value" (and not 0) so that negative maxima
      // are also recorded when all the points are used (sf <= 0).
      auto local_max_stress = no_stress_value;
      auto local_max_loc = std::array<mfem_mgis::real, 3u>{0.0, 0.0, 0.0};

      // bubble center coordinates
      const auto bx = b.center[0];
      const auto by = b.center[1];
      const auto bz = b.center[2];

      // X-axis slice:
      // Only search points within distance 'rs' along the X-axis
      const auto min_x_search = bx - rs;
      const auto max_x_search = bx + rs;

      // Find the starting point for our search using binary search on sorted
      // X-coordinates
      const auto* start_ptr = points_data;
      const auto* end_ptr = points_data + num_points;

      // Find first point with X-coordinate >= min_x_search
      const auto* it = std::lower_bound(start_ptr, end_ptr, min_x_search,
                                        [](const auto& point, double val) {
                                          return point.location[0] < val;
                                        });

      // Scan through candidate points, using early exits and filtering
      // Stop when we leave the X-slice
      for (; it != end_ptr; ++it) {
        const auto& pt = *it;

        // Early exit: If X-coordinate exceeds our search window, stop
        // immediately (points are sorted by X, so all remaining points are also
        // too far --> no need to do the rest of the stuff, we
        // dont care about the stress there!)
        if (pt.location[0] > max_x_search) break;
        // Now we carry on with a matrioska checking
        // Second check is on the y: check if point is within bounding box on
        // Y-axis
        if (std::abs(pt.location[1] - by) > rs) continue;

        // Third check is on the z: check if point is within bounding box on
        // 2-axis
        if (std::abs(pt.location[2] - bz) > rs) continue;
        // if we made here, the point is worthwile of consideration!
        const auto dx = pt.location[0] - bx;
        const auto dy = pt.location[1] - by;
        const auto dz = pt.location[2] - bz;

        const auto dist = [&dx, &dy, &dz] {
          return std::sqrt(tfel::math::power<2>(dx) + tfel::math::power<2>(dy) +
                           tfel::math::power<2>(dz));
        }();

        // If point is within spherical radius 'rs' and has higher stress than
        // current max, update the maximum stress and its location
        if ((dist < rs) && (local_max_stress < pt.value)) {
          local_max_stress = pt.value;
          local_max_loc = pt.location;
        }
      }

      // Store the maximum stress information found for this bubble
      results_data[i] = BubbleInfoRecord(b.boundary_identifier, local_max_loc,
                                         local_max_stress);
    });
  }

  // Combine the per-bubble maxima of all the MPI processes on process 0.
  // Each process sends [stress, x, y, z] for every bubble (a few tens of KB),
  // and process 0 keeps, for every bubble, the entry with the largest stress.
  // Since every integration point belongs to exactly one process, this is
  // the maximum over the whole neighbourhood of the bubble.
  {
    CatchTimeSection(ctx, "BubbleReduction");
    int nprocs = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const auto nb = bubbles_information.size();
    constexpr std::size_t values_per_bubble = 4;  // stress, x, y, z
    auto local_values =
        std::vector<mfem_mgis::real>(values_per_bubble * nb);
    for (std::size_t i = 0; i != nb; ++i) {
      const auto& bi = bubbles_information[i];
      local_values[values_per_bubble * i] = bi.stress_value;
      local_values[values_per_bubble * i + 1] = bi.location[0];
      local_values[values_per_bubble * i + 2] = bi.location[1];
      local_values[values_per_bubble * i + 3] = bi.location[2];
    }
    // only process 0 needs the receive buffer
    auto all_values = std::vector<mfem_mgis::real>(
        rank == 0 ? values_per_bubble * nb * nprocs : 0);
    const auto count = static_cast<int>(values_per_bubble * nb);
    MPI_Gather(local_values.data(), count, MPI_DOUBLE, all_values.data(),
               count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      for (int proc = 0; proc != nprocs; ++proc) {
        const auto* const v =
            all_values.data() + values_per_bubble * nb * proc;
        for (std::size_t i = 0; i != nb; ++i) {
          auto& bi = bubbles_information[i];
          if (v[values_per_bubble * i] > bi.stress_value) {
            bi.stress_value = v[values_per_bubble * i];
            bi.location = {v[values_per_bubble * i + 1],
                           v[values_per_bubble * i + 2],
                           v[values_per_bubble * i + 3]};
          }
        }
      }
    }
  }

  // Bubble results (CSV) and mean hydrostatic stress (sig_hydro.txt): always
  // computed and written, since they are cheap. In the original version they
  // were only written with `-p 1`, which also exports the full stress fields
  // for ParaView (tens of GB for big meshes).
  {
    CatchTimeSection(ctx, "common::bubble_results");

    if (mfem_mgis::isMPIMainProcessForDefaultCommunicator()) {
      // Bubbles with no integration point in their search sphere (or none
      // above the threshold): report them, they are written as "nan".
      auto nb_without_value = std::size_t{0};
      for (const auto& el : bubbles_information) {
        if (el.stress_value == no_stress_value) {
          ++nb_without_value;
        }
      }
      if (nb_without_value != 0) {
        // NB: Message adds a space between its arguments
        Message("WARNING:", nb_without_value, "bubble(s) out of",
                bubbles_information.size(),
                "have no value (no integration point found). Use -sf 0 "
                "to give a value to every bubble and/or -dl > 0 so that the "
                "search sphere is larger than the bubble.");
      }
      // Detailed log, only on demand (one block per bubble)
      if (p.verbosity_level > 0) {
        for (const auto& el : bubbles_information) {
          Message("Bubble ID = ", el.boundary_id);
          Message("Stress = ", el.stress_value);
          for (const auto& el1 : el.location) {
            Message("Location=", el1);
          }
        }
      }
      // Write the CSV file (header + one line per bubble)
      write_bubble_infos(output_file_locations, bubbles, bubbles_information);
    }

    if (mfem_mgis::isMPIMainProcessForDefaultCommunicator())
      write_message(outfile_sighydr, "Volume", "Integral", "Average");

    // Hydrostatic stress tr(sigma)/3 at the integration points, then its
    // average over the RVE (negative = compression, because the macroscopic
    // strain is imposed to zero)
    mfem_mgis::computeHydrostaticPressure(
        ctx, hyp, m, mfem_mgis::Material::END_OF_TIME_STEP) |
        or_die;
    calculateAverageHydrostaticStress<true>(
        mfem_mgis::may_abort, ctx, outfile_sighydr,
        problem.getImplementation<true>(), hyp);
  }

  // Fields for visualisation in ParaView: only with `-p 1`
  if (post_processing) {
    CatchTimeSection(ctx, "common::post_processing_step");

    // Execute standard post-processing (Paraview export of the stress).
    // NB: the return values below are checked with `or_die` so that a
    // failure prints its error message (the original code ignored them, and
    // a failed L2 projection then crashed with a segmentation fault).
    post_process(ctx, problem, 0, 1) | or_die;

    // First principal stress at the integration points
    auto end = mfem_mgis::Material::END_OF_TIME_STEP;
    mfem_mgis::computeFirstEigenStress(ctx, eig, m, end) | or_die;

    // L2 projection of the first eigenstress at the nodes
    // for better visualization
    auto& lsf = mfem_mgis::LinearSolverFactory<true>::getFactory();
    auto& fespace =
        problem.getFiniteElementDiscretization().getFiniteElementSpace<true>();
    auto linear_solver =
        lsf.generate(ctx, "HyprePCG", fespace,
                     mfem_mgis::Parameters{
                         {"Preconditioner",
                          mfem_mgis::Parameters{{"Name", "HypreILU"}}}}) |
        or_die;

    // empty result (e.g. linear solver not converged) -> error message
    auto projection =
        mfem_mgis::computeL2Projection<true>(ctx, linear_solver, {eig}) |
        or_die;

    auto eigenstress_gf_ptr = projection.result.get();

    mfem::ParaViewDataCollection exporter("Eigenstress-pp");
    exporter.SetDataFormat(mfem::VTKFormat::BINARY);
    exporter.RegisterField("FirstEigenStress", eigenstress_gf_ptr);
    exporter.SetMesh(
        &(problem.getFiniteElementDiscretization().getMesh<true>()));
    exporter.SetCycle(1);
    exporter.SetTime(1);
    exporter.Save();
  }

  // Clean up, close, and win
  if (mfem_mgis::isMPIMainProcessForDefaultCommunicator()) {
    outfile_sighydr.close();
    output_file_locations.close();
  }

  print_memory_footprint("[End]");
  mfem_mgis::Profiler::OutputManager::printTimeTable(ctx);
  return EXIT_SUCCESS;
}
