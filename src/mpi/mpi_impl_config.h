#ifndef MPI_IMPL_CONFIG_H
#define MPI_IMPL_CONFIG_H

// Define any MPI implementation-specific configurations here
#define RESULTS_PATH_MPI "/home/lesi-nedo/Desktop/master/first-year/second-semester/PDS/project/results/mpi"
#define FF_THREADS {std::make_tuple(1,1), std::make_tuple(2,2), std::make_tuple(4,2), std::make_tuple(8,4), std::make_tuple(8, 1), std::make_tuple(16,1), std::make_tuple(32,1)}

#endif // MPI_IMPL_CONFIG_H
