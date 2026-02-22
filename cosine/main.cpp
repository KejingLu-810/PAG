#include <stdlib.h>

void FastGraph(int efc_, int M_, int data_size_, int query_size_, int dim_, char* path_q_, char* path_data_, char* truth_data_, char* path_index_, int L_, int topk_);
int main(int argc, char** argv) {
    char* data_path = argv[1];
	char* query_path = argv[2];
	char* truth_path = argv[3];
	char* index_path = argv[4];	
	int vecsize = atoi(argv[5]);
	int qsize = atoi(argv[6]);
	int dim = atoi(argv[7]);
	int topk = atoi(argv[8]);
	
	int efc = atoi(argv[9]);
	int M = atoi(argv[10]);
	int L = atoi(argv[11]);

    FastGraph(efc, M, vecsize, qsize, dim, query_path, data_path, truth_path, index_path, L, topk);

    return 0;
};