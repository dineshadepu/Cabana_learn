#include <Cabana_Core.hpp>
#include <math.h>

#include <iostream>

#define DIM 2


//---------------------------------------------------------------------------//
// AoSoA example.
//---------------------------------------------------------------------------//
template<class MemorySpace, class ExecutionSpace>
void aosoaExample()
{
  /*
    Start by declaring the types in our tuples will store. Store a rank-2
    array of doubles, a rank-1 array of floats, and a single integer in
    each tuple.
  */
  using particle_members= Cabana::MemberTypes<double[3][3], float[4], int>;
  using particle_list = Cabana::AoSoA<particle_members, MemorySpace>;

  /*
    Next declare the vector length of our SoAs. This is how many tuples the
    SoAs will contain. A reasonable number for performance should be some
    multiple of the vector length on the machine you are using.
  */
  const int VectorLength = 8;

  /*
    Create the AoSoA. We define how many tuples the aosoa will
    contain. Note that if the number of tuples is not evenly divisible by
    the vector length then the last SoA in the AoSoA will not be entirely
    full (although its memory will still be allocated). The AoSoA label
    allows one to track the managed memory in an AoSoA through the Kokkos
    allocation tracker.
  */
  particle_list aosoa( "my_aosoa");
  // Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> aosoa( "my_aosoa");
   // int num_tuple = 5;
  // Cabana::AoSoA<DataTypes, MemorySpace> aosoa( "my_aosoa",
  // num_tuple);
  aosoa.resize( 100 );

  /*
    Print the label and size data. In this case we have created an AoSoA
    with 5 tuples. Because a vector length of 4 is used, a total memory
    capacity for 8 tuples will be allocated in 2 SoAs.
  */
  std::cout << "aosoa.label() = " << aosoa.label() << std::endl;
  std::cout << "aosoa.size() = " << aosoa.size() << std::endl;
  std::cout << "aosoa.capacity() = " << aosoa.capacity() << std::endl;
  std::cout << "aosoa.numSoA() = " << aosoa.numSoA() << std::endl;

}



int main( int argc, char* argv[] )
{

  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );
  // check inputs and write usage
  if ( argc < 1 )
    {
      std::cerr << "Usage: ./TwoBlocksColliding exec_space \n";

      std::cerr << "      exec_space      execute with: serial, openmp, "
	"cuda, hip\n";
      std::cerr << "\nfor example: ./TwoBlocksColliding serial\n";
      Kokkos::finalize();
      MPI_Finalize();
      return 0;
    }

  // execution space
  std::string exec_space( argv[1] );


  if ( 0 == exec_space.compare( "serial" ) ||
       0 == exec_space.compare( "Serial" ) ||
       0 == exec_space.compare( "SERIAL" ) )
    {
#ifdef KOKKOS_ENABLE_SERIAL
      aosoaExample<Kokkos::HostSpace, Kokkos::Serial>();
#else
      throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
  else if ( 0 == exec_space.compare( "openmp" ) ||
	    0 == exec_space.compare( "OpenMP" ) ||
	    0 == exec_space.compare( "OPENMP" ) )
    {
#ifdef KOKKOS_ENABLE_OPENMP
      aosoaExample<Kokkos::HostSpace, Kokkos::OpenMP>();
#else
      throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
  else if ( 0 == exec_space.compare( "cuda" ) ||
	    0 == exec_space.compare( "Cuda" ) ||
	    0 == exec_space.compare( "CUDA" ) )
    {
#ifdef KOKKOS_ENABLE_CUDA
      aosoaExample<Kokkos::CudaSpace, Kokkos::Cuda>();
#else
      throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }


  Kokkos::finalize();

  MPI_Finalize();
  return 0;
}
