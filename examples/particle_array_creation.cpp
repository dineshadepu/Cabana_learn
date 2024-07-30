#include <Cabana_Core.hpp>
#include <math.h>

#include <iostream>

#define DIM 2


//---------------------------------------------------------------------------//
// AoSoA example.
//---------------------------------------------------------------------------//
template <class MemorySpace, int Dimension>
class Particles
{
public:
  using memory_space = MemorySpace;
  using execution_space = typename memory_space::execution_space;
  static constexpr int dim = Dimension;

  // x, u, f (vector matching system dimension).
  using vector_type = Cabana::MemberTypes<double[dim]>;
  // volume, dilatation, weighted_volume.
  using scalar_type = Cabana::MemberTypes<double>;
  // no-fail.
  using int_type = Cabana::MemberTypes<int>;
  // type, W, v, rho, damage.
  using other_types =
    Cabana::MemberTypes<int, double, double[dim], double, double>;
  // Potentially needed later: body force (b), ID.

  // FIXME: add vector length.
  // FIXME: enable variable aosoa.
  using aosoa_u_type = Cabana::AoSoA<vector_type, memory_space, 1>;
  using aosoa_nofail_type = Cabana::AoSoA<int_type, memory_space, 1>;
  using aosoa_other_type = Cabana::AoSoA<other_types, memory_space>;

  // Constructor which initializes particles on regular grid.
  template <class ExecSpace>
  Particles( const ExecSpace& exec_space, std::size_t n )
  {
    resize( n );
    createParticles( exec_space );
  }

  // Constructor which initializes particles on regular grid.
  template <class ExecSpace>
  void createParticles( const ExecSpace& exec_space )
  {
    auto type = sliceType();
    auto u = sliceDisplacement();
    // u.resize(100);
    auto nofail = sliceNoFail();

    auto create_particles_func = KOKKOS_LAMBDA( const int i )
      {
	for (int j=0; j < DIM; j++){
	  u( i, j ) = DIM * i + j;
	}
      };
    Kokkos::RangePolicy<ExecSpace> policy( 0, u.size() );
    Kokkos::parallel_for( "create_particles_lambda", policy,
			  create_particles_func );
  }

  auto sliceDisplacement()
  {
    return Cabana::slice<0>( _aosoa_u, "displacements" );
  }
  auto sliceDisplacement() const
  {
    return Cabana::slice<0>( _aosoa_u, "displacements" );
  }
  auto sliceType() { return Cabana::slice<0>( _aosoa_other, "type" ); }
  auto sliceType() const { return Cabana::slice<0>( _aosoa_other, "type" ); }
  auto sliceStrainEnergy()
  {
    return Cabana::slice<1>( _aosoa_other, "strain_energy" );
  }
  auto sliceStrainEnergy() const
  {
    return Cabana::slice<1>( _aosoa_other, "strain_energy" );
  }
  auto sliceVelocity()
  {
    return Cabana::slice<2>( _aosoa_other, "velocities" );
  }
  auto sliceVelocity() const
  {
    return Cabana::slice<2>( _aosoa_other, "velocities" );
  }
  auto sliceDensity() { return Cabana::slice<3>( _aosoa_other, "density" ); }
  auto sliceDensity() const
  {
    return Cabana::slice<3>( _aosoa_other, "density" );
  }
  auto sliceDamage() { return Cabana::slice<4>( _aosoa_other, "damage" ); }
  auto sliceDamage() const
  {
    return Cabana::slice<4>( _aosoa_other, "damage" );
  }
  auto sliceNoFail()
  {
    return Cabana::slice<0>( _aosoa_nofail, "no_fail_region" );
  }
  auto sliceNoFail() const
  {
    return Cabana::slice<0>( _aosoa_nofail, "no_fail_region" );
  }

  void resize(const std::size_t n)
  {
    _aosoa_u.resize( n );
    _aosoa_nofail.resize( n );
    _aosoa_other.resize( n );
  }


private:
  aosoa_u_type _aosoa_u;
  aosoa_nofail_type _aosoa_nofail;
  aosoa_other_type _aosoa_other;

  // Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
};

template<class MemorySpace, class ExecutionSpace>
void aosoaExample(const ExecutionSpace& exec_space)
{

  auto particles = Particles<MemorySpace, DIM>(exec_space, 100);

  /*
    Print the label and size data. In this case we have created an AoSoA
    with 5 tuples. Because a vector length of 4 is used, a total memory
    capacity for 8 tuples will be allocated in 2 SoAs.
  */
  std::cout << "aosoa.label() = " << particles.sliceNoFail().label() << std::endl;
  std::cout << "aosoa.label() = " << particles.sliceDensity().label() << std::endl;
  std::cout << "aosoa.label() = " << particles.sliceDamage().label() << std::endl;
  std::cout << "aosoa.label() = " << particles.sliceVelocity().label() << std::endl;

  auto u = particles.sliceDisplacement();
    for (int i=0; i < u.size(); i++){
      std::cout << u (i, 0) << "\n";
    }
  // std::cout << "aosoa.label() = " <<  << std::endl;
  // std::cout << "aosoa.size() = " << particles.size() << std::endl;
  // std::cout << "aosoa.capacity() = " << particles.capacity() << std::endl;
  // std::cout << "aosoa.numSoA() = " << particles.numSoA() << std::endl;

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
      aosoaExample<Kokkos::HostSpace, Kokkos::Serial>(Kokkos::Serial());
#else
      throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
  else if ( 0 == exec_space.compare( "openmp" ) ||
	    0 == exec_space.compare( "OpenMP" ) ||
	    0 == exec_space.compare( "OPENMP" ) )
    {
#ifdef KOKKOS_ENABLE_OPENMP
      aosoaExample<Kokkos::HostSpace, Kokkos::OpenMP>(Kokkos::OpenMP());
#else
      throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
  else if ( 0 == exec_space.compare( "cuda" ) ||
	    0 == exec_space.compare( "Cuda" ) ||
	    0 == exec_space.compare( "CUDA" ) )
    {
#ifdef KOKKOS_ENABLE_CUDA
      aosoaExample<Kokkos::CudaSpace, Kokkos::Cuda>(Kokkos::Cuda());
#else
      throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }


  Kokkos::finalize();

  MPI_Finalize();
  return 0;
}
