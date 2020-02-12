#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define DEBUG 0 /* enable to display debug information */
#define CHECK 0 /* enable to test the parallel algorithm with the serial */

/*
 * used links:
 * https://parallel.ru/vvv/mpi.html
 * https://www.mpich.org/static/docs/v3.1/www3/MPI_Type_create_subarray.html
 * https://www.mpich.org/static/docs/v3.1/www3/MPI_Type_create_resized.html
 * https://www.mpich.org/static/docs/v3.1/www3/MPI_Type_commit.html
 * https://www.mpich.org/static/docs/v3.1/www3/MPI_Scatterv.html
 * https://www.mpich.org/static/docs/v3.1/www3/MPI_Barrier.html
 * https://www.mpich.org/static/docs/v3.1/www3/MPI_Bcast.html
 * https://www.mpich.org/static/docs/v3.1/www3/MPI_Sendrecv_replace.html
 * https://www.mpich.org/static/docs/v3.1/www3/MPI_Gatherv.html
 *
 */

typedef struct {
    MPI_Comm grid_comm; /* handle to global grid communicator */
    MPI_Comm row_comm;  /* row communicator */
    MPI_Comm col_comm;  /* column communicator */
    int n_proc;         /* number of processors */
    int grid_dim;       /* dimension of the grid, = sqrt(n_proc) */
    int my_row;         /* row position of a processor in a grid */
    int my_col;         /* column position of a procesor in a grid */
    int my_rank;        /* rank within the grid */
} GridInfo;


/**
 * [FoxAlgorithm: parallel fox matrix multiplication algorithm]
 */
void FoxAlgorithm(double *A, double *B, double *C, int size, GridInfo *grid);

/**
 * [grid_init: process grid initialization]
 */
void grid_init(GridInfo *grid);

/**
 * [matrix functions]
 */
void matrix_creation(double **pA, double **pB, double **pC, int size);
void matrix_removal(double **pA, double **pB, double **pC);
void matrix_init(double *A, double *B, int size, int sup);
void matrix_dot(double *A, double *B, double *C, int n);
int  matrix_check(double *A, double *B, int n);
void matrix_print(double *A, int n);



void grid_init(GridInfo *grid)
{
    int old_rank;
    int dimensions[2];
    int wrap_around[2];
    int coordinates[2];
    int free_coords[2];

    /* get the overall information before overlaying cart_grid */
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->n_proc));
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

    grid->grid_dim = (int)sqrt(grid->n_proc);
    /* ERROR check: */
    if (grid->grid_dim * grid->grid_dim != grid->n_proc) {
        printf("[!] \'-np\' is a perfect square!\n");
        exit(-1);
    }
    /* set the dimensions */
    dimensions[0] = dimensions[1] = grid->grid_dim;
    wrap_around[0] = wrap_around[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1, &(grid->grid_comm));
    /* since we have set reorder to true, this might have changed the ranks */
    MPI_Comm_rank(grid->grid_comm, &(grid->my_rank));
    /* get the cartesian coordinates for the current process */
    MPI_Cart_coords(grid->grid_comm, grid->my_rank, 2, coordinates);
    /* set the coordinate values for the current process */
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    /* create row communicators */
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->row_comm));

    /* create column communicators */
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->col_comm));
}


void matrix_creation(double **pA, double **pB, double **pC, int size)
{
    *pA = (double *)malloc(size * size * sizeof(double));
    *pB = (double *)malloc(size * size * sizeof(double));
    *pC = (double *)calloc(size * size, sizeof(double));
}


void matrix_init(double *A, double *B, int size, int sup)
{
    srand(time(NULL));
    for (int i = 0; i < size * size; ++i) {
        *(A + i) = rand() % sup + 1;
        *(B + i) = rand() % sup + 1;
    }
}


void matrix_dot(double *A, double *B, double *C, int size)
{
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}


int matrix_check(double *A, double *B, int size)
{
    for (int i = 0; i < size * size; ++i) {
        if (*(A + i) != *(B + i)) {
            return 0;
        }
    }
    return 1;
}


void matrix_print(double *A, int size)
{
    printf("---~---~---~---~---~---~---~---\n");
    for (int i = 0; i < size * size; ++i) {
        printf("%.2lf ", *(A + i));
        if ((i + 1) % size == 0){
            printf("\n");
        }
    }
    printf("---~---~---~---~---~---~---~---\n\n");
}


void matrix_removal(double **pA, double **pB, double **pC)
{
    free(*pA);
    free(*pB);
    free(*pC);
}


/* ------------------------------ FoxAlgorithm ------------------------------ */

void FoxAlgorithm(double *A, double *B, double *C, int size, GridInfo *grid)
{
    double *buff_A = (double*)calloc(size * size, sizeof(double));
    MPI_Status status;
    int root;
    int src = (grid->my_row + 1) % grid->grid_dim;
    int dst = (grid->my_row -1 + grid->grid_dim) % grid->grid_dim;

    /**
     * For each iterations:
     *   1. find the blocks that are forming a diagonal
     *   2. shared that block on the row it belongs to
     *   3. multiply the updated A (or buff_A) with B onto C
     *   4. shift the B blocks upward
     */
    for (int stage = 0; stage < grid->grid_dim; ++stage) {
        root = (grid->my_row + stage) % grid->grid_dim;
        if (root == grid->my_col) {
            MPI_Bcast(A, size * size, MPI_DOUBLE, root, grid->row_comm);
            matrix_dot(A, B, C, size);
        } else {
            MPI_Bcast(buff_A, size * size, MPI_DOUBLE, root, grid->row_comm);
            matrix_dot(buff_A, B, C, size);
        }
        MPI_Sendrecv_replace(B, size * size, MPI_DOUBLE, dst, 0, src, 0, grid->col_comm, &status);
    }
}

/* -------------------------------------------------------------------------- */


int main(int argc, char **argv)
{
    double *pA, *pB, *pC;
    double *local_pA, *local_pB, *local_pC;
    int matrix_size = 100;
    if (argc == 2) {
        sscanf(argv[1], "%d", &matrix_size);
    }
    /* -------------------------------------------------- */
    MPI_Init(&argc, &argv);
    /* -------------------------------------------------- */
    GridInfo grid;
    grid_init(&grid);
    /* ERROR check: */
    if (matrix_size % grid.grid_dim != 0) {
        printf("[!] matrix_size mod sqrt(n_processes) != 0 !\n");
        exit(-1);
    }
    /* -------------------------------------------------- */
    if (grid.my_rank == 0) {
        matrix_creation(&pA, &pB, &pC, matrix_size);
        matrix_init(pA, pB, matrix_size, 10);
#if DEBUG
        printf("pA (size=%d):\n", matrix_size); matrix_print(pA, matrix_size);
        printf("pB (size=%d):\n", matrix_size); matrix_print(pB, matrix_size);
        // printf("pC (size=%d):\n", matrix_size); matrix_print(pC, matrix_size);
#endif
    }
    int local_matrix_size = matrix_size / grid.grid_dim;
    matrix_creation(&local_pA, &local_pB, &local_pC, local_matrix_size);
    /* -------------------------------------------------- */
    MPI_Datatype blocktype, type;
  	int array_size[2] = {matrix_size, matrix_size};
  	int subarray_sizes[2] = {local_matrix_size, local_matrix_size};
  	int array_start[2] = {0, 0};
  	MPI_Type_create_subarray(2, array_size, subarray_sizes, array_start,
                             MPI_ORDER_C, MPI_DOUBLE, &blocktype);
  	MPI_Type_create_resized(blocktype, 0, local_matrix_size * sizeof(double), &type);
  	MPI_Type_commit(&type);
    /* -------------------------------------------------- */
  	int displs[grid.n_proc];
  	int sendcounts[grid.n_proc];
    if (grid.my_rank == 0) {
        for (int i = 0; i < grid.n_proc; ++i) {
            sendcounts[i] = 1;
        }
        int disp = 0;
        for (int i = 0; i < grid.grid_dim; ++i) {
          	for (int j = 0; j < grid.grid_dim; ++j) {
            		displs[i * grid.grid_dim + j] = disp;
            		disp += 1;
            }
            disp += (local_matrix_size - 1) * grid.grid_dim;
        }
    }
    /* -------------------------------------------------- */
    MPI_Scatterv(pA, sendcounts, displs, type, local_pA,
                 local_matrix_size * local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  	MPI_Scatterv(pB, sendcounts, displs, type, local_pB,
                 local_matrix_size * local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* -------------------------------------------------- */
#if DEBUG
    int print_rank = 0;
    while (print_rank < grid.n_proc) {
         if (grid.my_rank == print_rank) {
             printf("%d. local_pA (size=%d):\n", grid.my_rank, local_matrix_size);
             matrix_print(local_pA, local_matrix_size);
             printf("%d. local_pB (size=%d):\n", grid.my_rank, local_matrix_size);
             matrix_print(local_pB, local_matrix_size);
         }
         print_rank++;
         MPI_Barrier(grid.grid_comm);
    }
#endif
    /* -------------------------------------------------- */
    double start_time, end_time;
    MPI_Barrier(grid.grid_comm);
    if (grid.my_rank == 0) {
        start_time = MPI_Wtime();
    }
    /* vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv */
    /* -------------------------------------------------- */
    FoxAlgorithm(local_pA, local_pB, local_pC, local_matrix_size, &grid);
    /* -------------------------------------------------- */
    /* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ */
    MPI_Barrier(grid.grid_comm);
    if (grid.my_rank == 0) {
        end_time = MPI_Wtime() - start_time;
    }
    /* collect submatrices from all processes: */
    MPI_Gatherv(local_pC, local_matrix_size*local_matrix_size, MPI_DOUBLE, pC, sendcounts, displs, type, 0, MPI_COMM_WORLD);
    /* -------------------------------------------------- */
    if (grid.my_rank == 0) {
        // FoxAlgorithm_time | number_processes | matrix_size"
        printf("%.5lf, %d, %d\n", end_time, grid.n_proc, matrix_size);
#if DEBUG
        /* Sequential implementation comparison */
        printf("pC:\n"); matrix_print(pC, matrix_size);
        double *pD = (double *)calloc(matrix_size * matrix_size, sizeof(double));
        start_time = MPI_Wtime();
        matrix_dot(pA, pB, pD, matrix_size);
        end_time = MPI_Wtime() - start_time;
        printf("pD:\n"); matrix_print(pD, matrix_size);
        printf("dot_time: %.5lf\n", end_time);
        printf("matrix_check: %s\n", matrix_check(pC, pD, matrix_size) ? "yes" : "NO");
        free(pD);
#endif
#if CHECK
        double *pD = (double *)calloc(matrix_size * matrix_size, sizeof(double));
        start_time = MPI_Wtime();
        matrix_dot(pA, pB, pD, matrix_size);
        end_time = MPI_Wtime() - start_time;
        printf("dot_time: %.5lf\n", end_time);
        printf("matrix_check: %s\n", matrix_check(pC, pD, matrix_size) ? "yes" : "NO");
        free(pD);
#endif
        matrix_removal(&pA, &pB, &pC);
    }
    matrix_removal(&local_pA, &local_pB, &local_pC);
    /* -------------------------------------------------- */
    MPI_Finalize();
    /* -------------------------------------------------- */
    return 0;
}
