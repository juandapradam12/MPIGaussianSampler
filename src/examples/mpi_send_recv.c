#include <mpi.h>
#include <stdio.h>

/* Point-to-point transfer of a single integer from rank 0 to rank 1. */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int number = 0;

    if (rank == 0) {
        number = 123;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Rank 0 sent %d to rank 1\n", number);
    } else if (rank == 1) {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 1 received %d from rank 0\n", number);
    }

    MPI_Finalize();
    return 0;
}
