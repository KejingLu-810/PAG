#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <time.h>
#include "hnswlib/paglib.h"
#include <omp.h>
#include <unordered_set>
#include <filesystem>
#define ALIGNMENT 64

using namespace std;
using namespace hnswlib;
namespace fs = std::filesystem;

static thread_local std::mt19937 rng(std::random_device{}());
static thread_local std::normal_distribution<double> nd(0.0, 1.0);

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }

};

void random_orthogonal_matrix(int subdim, std::mt19937 &rng, std::vector<std::vector<float>> &R) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    R.assign(subdim, std::vector<float>(subdim, 0.0f));

    for (int i = 0; i < subdim; i++)
        for (int j = 0; j < subdim; j++)
            R[i][j] = nd(rng);

    for (int i = 0; i < subdim; i++) {
        for (int j = 0; j < i; j++) {
            float dot = 0.0f;
            for (int k = 0; k < subdim; k++)
                dot += R[i][k] * R[j][k];
            for (int k = 0; k < subdim; k++)
                R[i][k] -= dot * R[j][k];
        }
        float norm = 0.0f;
        for (int k = 0; k < subdim; k++)
            norm += R[i][k] * R[i][k];
        norm = std::sqrt(norm);
        for (int k = 0; k < subdim; k++)
            R[i][k] /= norm;
    }
}

void shuffle_for_equal_norm(std::vector<float>& dim_norm, int vecsize, int vecdim, int level,
                            std::vector<int>& permutation, std::vector<int>& zero_positions) {
    permutation.resize(vecdim);
	
	zero_positions.resize(level); 
	
    std::vector<int> idx(vecdim);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&dim_norm](int a, int b) {
        return dim_norm[a] > dim_norm[b];
    });

    std::vector<float> seg_norm(level, 0.0f);
	std::vector<int> seg_size(level, 0);
    std::vector<std::vector<int>> segments(level);

    int K = vecdim / level;
    for (int k = 0; k < vecdim; k++) {
        int dim_id = idx[k];
    
        int best_seg = -1;
        float best_norm = std::numeric_limits<float>::max();
    
        for (int s = 0; s < level; s++) {
            if (seg_size[s] < K && seg_norm[s] < best_norm) {
                best_norm = seg_norm[s];
                best_seg = s;
            }
        }

        segments[best_seg].push_back(dim_id);
        seg_norm[best_seg] += dim_norm[dim_id];
        seg_size[best_seg]++;
    }

    for (int l = 0; l < level; l++) {
        int subdim = segments[l].size();
        int subdim0 = 0;

        for (int k = subdim - 1; k >= 0; k--) {
            int dim_id = segments[l][k];
            if (dim_norm[dim_id] > 0.0f) {
                subdim0 = k + 1;
                break;
            }
        }
        zero_positions[l] = subdim0;
    }

    int pos = 0;
    for (int l = 0; l < level; l++)
        for (int d : segments[l])
            permutation[pos++] = d;
}

void generate_subspace_vectors_projvec(std::vector<std::vector<std::vector<float>>> &projVec,
                                       int level, int subdim, int m, std::vector<int>& zero_positions) {
    std::normal_distribution<float> nd(0.0f, 1.0f);

#pragma omp parallel for
for(int l = 0; l < level; l++){
    std::random_device rd;
    std::mt19937 rng_thread(rd() + l);

    int subdim0 = zero_positions[l];   
    if (subdim0 <= 0) subdim0 = subdim;
	
    std::vector<std::vector<float>> vectors(m, std::vector<float>(subdim0, 0.0f));
    if (m <= subdim0) {    
        for (int i = 0; i < m; i++)
            vectors[i][i] = 1.0f;

        std::vector<std::vector<float>> R;
        random_orthogonal_matrix(subdim0, rng_thread, R);
        for (int i = 0; i < m; i++) {
            std::vector<float> tmp(subdim0, 0.0f);
            for (int r = 0; r < subdim0; r++)
                for (int c = 0; c < subdim0; c++)
                    tmp[r] += R[r][c] * vectors[i][c];
            vectors[i] = tmp;
        }
    } else {
        int n_poly = m / subdim0;
        int remainder = m % subdim0;
        int idx = 0;

        for (int p = 0; p < n_poly; p++) {
            for (int i = 0; i < subdim0; i++) {
                std::fill(vectors[idx+i].begin(), vectors[idx+i].end(), 0.0f);
                vectors[idx+i][i] = 1.0f;
            }

            std::vector<std::vector<float>> R;
            random_orthogonal_matrix(subdim0, rng_thread, R);
            for (int i = 0; i < subdim0; i++) {
                std::vector<float> tmp(subdim0, 0.0f);
                for (int r = 0; r < subdim0; r++)
                    for (int c = 0; c < subdim0; c++)
                        tmp[r] += R[r][c] * vectors[idx+i][c];
                vectors[idx+i] = tmp;
            }
            idx += subdim0;
        }
		
        if (remainder > 0) {
            for (int i = 0; i < remainder; i++) {
                std::fill(vectors[idx+i].begin(), vectors[idx+i].end(), 0.0f);
                vectors[idx+i][i] = 1.0f;
            }
			
            std::vector<std::vector<float>> R;
            random_orthogonal_matrix(subdim0, rng_thread, R);
            for (int i = idx; i < idx + remainder; i++) {
                std::vector<float> tmp(subdim0, 0.0f);
                for (int r = 0; r < subdim0; r++)
                    for (int c = 0; c < subdim0; c++)
                        tmp[r] += R[r][c] * vectors[i][c];
            
			    vectors[i] = tmp;
            }
        }	
    }

    float scale = 1.0f / std::sqrt((float)level);
        for (int i = 0; i < m; i++){
            for (int j = 0; j < subdim0; j++)
                projVec[l][i][j] = vectors[i][j] * scale;

            for (int j = subdim0; j < subdim; j++)
                projVec[l][i][j] = 0;
	    }
    }
}

static void
get_gt(unsigned int *massQA, size_t qsize, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, size_t maxk) {
    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[maxk * i + j]);
        }
    }
}

static void
test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, float* table, size_t vecdim, size_t extended_dim, char* path_index) {
    vector<size_t> efs;// = { 10,10,10,10,10 };

	size_t padded_dim = (vecdim + 15) & ~0xF;
    std::vector<int> permutation(extended_dim);

    std::string folderPath(path_index);
    std::string fullPath;
    if (!folderPath.empty() && (folderPath.back() == '/' || folderPath.back() == '\\')) {
        fullPath = folderPath + "permutation.bin";
    } else {
        fullPath = folderPath + "/permutation.bin";
    }

    std::ifstream fin(fullPath, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open permutation file!" << std::endl;
    }
	fin.read((char*)permutation.data(), permutation.size() * sizeof(int));

    float* permutedQ = new (std::align_val_t{ALIGNMENT}) float[qsize * extended_dim]; 
	
	int step;
	
	if(k <= 10)
		step = 10;
	else if(k <= 100)
		step = 100;
	else
		step = k;

    std::vector<std::vector<Neighbor>> result; 
    result.resize(qsize);
    for (size_t i = 0; i < qsize; ++i) {
        result[i].resize(step);
    }
    if (k == 10) {
        #ifndef DEBUG_MODE
        for (int i = step; i < 500; i+=step) efs.push_back(i);
        #else
        // Target range [500, 5000], must be multiples of 10
        // 1. Initial phase: Densely capture variations (500 - 1000)
        for (int i = 500; i <= 1000; i += 20) efs.push_back(i);
        // 2. Middle phase: Medium density (1100 - 3000)
        for (int i = 1100; i <= 3000; i += 100) efs.push_back(i);
        // 3. Final phase: High density to capture long tail (3200 - 5000)
        // for (int i = 3200; i <= 5000; i += 200) efs.push_back(i);
        #endif
    } else if (k == 100) {
        #ifndef DEBUG_MODE
        for (int i = step; i < 1500; i+=step) efs.push_back(i);
        #else
        // Target range [1500, 50000], must be multiples of 100
        // 1. Initial phase: Ultra-high density (1500 - 5000)
        for (int i = 1500; i <= 5000; i += 100) efs.push_back(i);
        // 2. Middle transition phase: (5500 - 10000)
        for (int i = 5500; i <= 10000; i += 500) efs.push_back(i);
        // 3. Long-tail convergence phase: Supplement points for high recall regions (16000 - 50000)
        // for (int i = 16000; i <= 50000; i += 1000) efs.push_back(i);
        #endif

    } else if (k == 1000) {
        #ifndef DEBUG_MODE
        for (int i = step; i < 5000; i+=step) efs.push_back(i);
        #else
        // Target range [5000, 200000], must be multiples of 1000
        // 1. Initial phase: Capture Recall rising phase (5000 - 10000)
        for (int i = 5000; i <= 10000; i += 1000) efs.push_back(i);
        // 2. Middle stable phase: (35000 - 100000)
        // for (int i = 35000; i <= 100000; i += 2000) efs.push_back(i);
        // 3. Ultra-high Recall region: (110000 - 200000)
        // for (int i = 110000; i <= 200000; i += 10000) efs.push_back(i);
        #endif
    }
	float prev_recall = -1.0f;
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        for (int q = 0; q < qsize; q++) {
            float* curQ = massQ + q * padded_dim;
            float* curP = permutedQ + q * extended_dim;

            for (int i = 0; i < extended_dim; i++){
				int new_pos = permutation[i];
                if(new_pos < padded_dim)
				    curP[i] = curQ[new_pos];
				else
					curP[i] = 0.0f;
			}
        }

        for (int i = 0; i < qsize; i++) {
            float* query_org = massQ + i * padded_dim;		
		    float* query_extended = permutedQ + extended_dim * i;
            appr_alg.searchKnn(query_org, query_extended, k, result[i], table, step);
        }
	    float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;	

        size_t correct = 0;
        size_t total = 0;		
		for (int i = 0; i < qsize; i++) { 
		    std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
            unordered_set<labeltype> g;
            total += gt.size();

            while (gt.size()) {

                g.insert(gt.top().second);
                gt.pop();
            }

            for(int j = 0; j < k; j++) {
                if (g.find(result[i][j].id) != g.end()) {
                    correct++;
                } 
            }
		}
		
        float recall = 1.0f * correct / total;
        float qps = 1e6f / time_us_per_query;
        cout << ef << "\t" << recall << "\t" << qps << " QPS\n";
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
        // if (prev_recall == recall) {
        //     cout << "[Info] Stopping: Recall improvement below threshold\n";
        //     break;
        // }
        prev_recall = recall;
        if (qps < 15.0f) {
            cout << "[Info] Stopping: QPS dropped below 25.0\n";
            break;
        }
    }
		
	delete[] permutedQ;
}

//inline bool exists_test(const std::string &name) {
//    ifstream f(name.c_str());
//    return f.good();
//}

void FastGraph(int efc_, int M_, int data_size_, int query_size_, int dim_, char* path_q_, char* path_data_, char* truth_data_, char* path_index_, int L_, int topk_) {

	int efConstruction = efc_;
	int M = M_;
	int step;
    if (efConstruction <= 2*M){
        efConstruction = 2*M;
		step = 2*M;
	}
	else if(efConstruction < 100 && efConstruction % (2*M) == 0){
	    step = 2*M;
	}
    else{
        efConstruction = (efConstruction + 99) / 100 * 100;
	    step = 100;
	}
	
    size_t maxk = 1000;
    size_t vecsize = data_size_;
    size_t qsize = query_size_;
    size_t vecdim = dim_;
	size_t padded_dim = (vecdim + 15) & ~0xF;   
    size_t extended_dim = ((padded_dim + L_ - 1) / L_) * L_;   

    //char path_index[1024];
    char *path_q = path_q_;
    char *path_data = path_data_;
    char *path_index = path_index_;
    //sprintf(path_index, "info.bin");

    int m = 8;
	int level = L_;
    int subdim = extended_dim / level;

    std::vector<std::vector<std::vector<float>>> projVec(
        level, std::vector<std::vector<float>>(
               m, std::vector<float>(subdim, 0.0f)));
	
    L2Space l2space(padded_dim);
	InnerProductSpace ipsubspace(subdim);
	InnerProductSpace ipspace(padded_dim);  
    HierarchicalNSW<float> *appr_alg;

    //int in = 0;
    std::vector<float> buf(vecdim);
    fs::path dir(path_index_); 	
    #ifdef DEBUG_MODE
    cout << "[Info]DEBUG_MODE is enable\n";
    #endif
    if (fs::exists(dir)) {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, &ipspace, &ipsubspace, path_index, false);

        cout << "Loading GT:\n";
        ifstream inputGT(truth_data_, ios::binary);
        unsigned int rows;
		unsigned int cols;
        inputGT.read((char*)&rows, 4);
        inputGT.read((char*)&cols, 4);

        if (rows != (unsigned int)qsize) {
            printf("Warning: Ground truth dimensions mismatch!\n");
        }

        unsigned int *massQA = new unsigned int[qsize * maxk];
        inputGT.read((char *)massQA, (size_t)qsize * maxk * 4);
        inputGT.close();
	
        cout << "Loading queries:\n";
	
        float* massQ = (float*) std::aligned_alloc(ALIGNMENT, qsize * padded_dim * sizeof(float));	
        ifstream inputQ(path_q, ios::binary);
        
        inputQ.read((char*)&rows, 4);
        inputQ.read((char*)&cols, 4);			
		for (int i = 0; i < qsize; i++) {
            //inputQ.read((char*)&in, 4);
            inputQ.read((char*)buf.data(), 4 * vecdim);
            
			float sum = 0;
			for(int j = 0; j < vecdim; j++){
				sum += buf[j]*buf[j];
			}
			sum = sqrt(sum);
			for(int j = 0; j < vecdim; j++)
				buf[j] /= sum;
			
			float* dst = &massQ[i * padded_dim];

            for (int j = 0; j < vecdim; j++)
                dst[j] = buf[j];

            for (int j = vecdim; j < padded_dim; j++)
                dst[j] = 0.0f;
        }	
        inputQ.close();	
	
        vector<std::priority_queue<std::pair<float, labeltype >>> answers;
        size_t k = topk_;
        get_gt(massQA, qsize, answers, k, maxk);

        const int TOTAL_ELEMENTS = level * 2 * m; 
        float* table = (float*) std::aligned_alloc(ALIGNMENT, TOTAL_ELEMENTS * sizeof(float));
        test_vs_recall(massQ, vecsize, qsize, *appr_alg, answers, k, table, vecdim, extended_dim, path_index);
    } else {
		fs::create_directories(dir);		
        cout << "Building FastGraph index:\n";	

        StopW stopw = StopW();
        StopW stopw_full = StopW();
        ifstream input(path_data, ios::binary);		
        float* vec = new float[extended_dim];	    
        std::vector<float> norm(vecsize, 0.0f);
        std::vector<float> center_data(padded_dim, 0.0f);
        std::vector<double> center_data2(padded_dim, 0.0f);
        std::vector<float> dim_norm(extended_dim, 0.0f);
        std::vector<double> dim_norm2(extended_dim, 0.0f);

        double temp; 
		
        uint32_t temp_npts = 0;
        uint32_t temp_dim = 0;

        input.read((char*)&temp_npts, sizeof(uint32_t));
        input.read((char*)&temp_dim, sizeof(uint32_t));	
		
		for(int i = 0; i < vecsize; i++){
            input.read((char*)buf.data(), 4 * vecdim);
            memcpy(vec, buf.data(), sizeof(float) * vecdim);
            memset(vec + vecdim, 0, sizeof(float) * (extended_dim - vecdim));			

			float sum = 0;
			for(int j = 0; j < vecdim; j++){
				sum += (vec[j] * vec[j]);
			}
			sum = sqrt(sum);
			norm[i] = sum;
			for(int j = 0; j < vecdim; j++){
				vec[j] = vec[j] / sum;
			}

            for (int d = 0; d < padded_dim; d++)
                center_data2[d] += vec[d];						
						
			for(int j = 0; j < vecdim; j++){
				dim_norm2[j] += vec[j] * vec[j];
			}
		}

        for (int d = 0; d < vecdim; d++)
            dim_norm[d] = dim_norm2[d] / vecsize;

        for (int d = 0; d < padded_dim; d++)
            center_data[d] = center_data2[d] / vecsize;

        float center_norm = 0;
        for (int d = 0; d < padded_dim; d++)
            center_norm += center_data[d] * center_data[d];
        
		center_norm = sqrt(center_norm);
        for (int d = 0; d < padded_dim; d++)
            center_data[d] /= center_norm;       

        std::vector<std::pair<float, int>> array(vecsize);
        std::vector<bool> skip_list(vecsize, false);
		
        input.clear();
        input.seekg(0, std::ios::beg);

        input.read((char*)&temp_npts, sizeof(uint32_t));
        input.read((char*)&temp_dim, sizeof(uint32_t));	
		
        for(int i = 0; i < vecsize; i++){
            input.read((char*)buf.data(), sizeof(float) * vecdim);

            float sum = 0;
            for(int j = 0; j < vecdim; j++){
                float tmp = buf[j] / norm[i] - center_data[j];
                sum += tmp * tmp;
            }
    
            array[i].first = sum;  
            array[i].second = i;
        }

        std::nth_element(array.begin(), array.begin() + step, array.end());
        std::sort(array.begin(), array.begin() + step);

        std::vector<std::pair<float, int>> sorted_targets(array.begin(), array.begin() + step);
        std::sort(sorted_targets.begin(), sorted_targets.end(), [](const auto& a, const auto& b) {
            return a.second < b.second; 
        });

        std::vector<std::vector<float>> result_vectors(step, std::vector<float>(vecdim));

        for(int i = 0; i < step; i++) {
            int target_id = sorted_targets[i].second;
            skip_list[target_id] = true;

            float cur_norm = norm[target_id];

            std::streampos absolute_offset = 8LL + static_cast<long long>(target_id) * (static_cast<long long>(vecdim) * 4);
            input.seekg(absolute_offset, std::ios::beg);
            input.read((char*)result_vectors[i].data(), 4 * vecdim);
            //std::streampos offset = static_cast<long long>(target_id) * (4 + 4 * vecdim);
            //input.seekg(offset + 4LL);   
            //input.read((char*)result_vectors[i].data(), 4 * vecdim);
			for(int j = 0; j < vecdim; j++)
				result_vectors[i][j] /= cur_norm;			
        }

        input.clear();
        input.seekg(0, std::ios::beg);

        input.read((char*)&temp_npts, sizeof(uint32_t));
        input.read((char*)&temp_dim, sizeof(uint32_t));	

        std::mt19937 rng(12345); 
        std::vector<int> permutation;
		std::vector<int> zero_positions;
		
		shuffle_for_equal_norm(dim_norm, vecsize, extended_dim, level, permutation, zero_positions);		
		generate_subspace_vectors_projvec(projVec, level, subdim, m, zero_positions);		
	
        std::string folderPath(path_index);
        std::string fullPath;
        if (!folderPath.empty() && (folderPath.back() == '/' || folderPath.back() == '\\')) {
            fullPath = folderPath + "permutation.bin";
        } else {
            fullPath = folderPath + "/permutation.bin";
        }
	
		std::ofstream fout(fullPath, std::ios::binary);
        fout.write((char*)permutation.data(), permutation.size() * sizeof(int));
        fout.close();

        std::vector<std::pair<float, int>>().swap(array);

        appr_alg = new HierarchicalNSW<float>(maxk, step, m, vecdim, projVec, level, subdim, permutation, &l2space, &ipspace, &ipsubspace, vecsize, path_index, M, efConstruction);

        for (int i = 0; i < step; i++) {
            float* mass = new float[extended_dim];
            std::memset(mass, 0, extended_dim * sizeof(float));
            std::memcpy(mass, result_vectors[i].data(), vecdim*sizeof(float));
			
            appr_alg->addPoint((void *) mass, (size_t) sorted_targets[i].second);
            delete[] mass;
		}		
/*		
		int j1 = -1;
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < vecsize; i++) {			
            int j2 = 0;			
            float* mass = new float[extended_dim];
            std::memset(mass, 0, extended_dim * sizeof(float));
            int tmp_in = 0;				
#pragma omp critical
            {
                input.read((char*)&tmp_in, 4);
                input.read((char*)mass, 4 * vecdim);				
				
                j1++;
                j2=j1;
            }
			
			if(skip_list[j2] == false){
			    for(int j = 0; j < vecdim; j++){
				    mass[j] /= norm[j2];
			    }
                appr_alg->addPoint((void *) mass, (size_t) j2);
			}
			delete[] mass;
        }
*/
//	    size_t report_every = 10000;		
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < vecsize; i++) {
    if (skip_list[i]) continue; 

    float* mass = new float[extended_dim];

    std::memset(mass, 0, extended_dim * sizeof(float));
    
    std::streampos absolute_offset = 8LL + static_cast<long long>(i) * (static_cast<long long>(vecdim) * 4);

    #pragma omp critical
    {
        input.seekg(absolute_offset, std::ios::beg);
        input.read((char*)mass, 4 * vecdim);

        //static int progress_count = 0; 
        //progress_count++;
        //if (progress_count % report_every == 0) {
        //    cout << progress_count / (0.01 * vecsize) << " %, "
        //         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips\n";
        //    stopw.reset();
        //}
    }

    for(int j = 0; j < vecdim; j++){
		mass[j] /= norm[i];
    }

    appr_alg->addPoint((void *) mass, (size_t) i);
    delete[] mass;
}

#ifndef WITHOUT_PES
        printf("PAG with PES optimization enabled.\n");
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < vecsize; i++) {
            appr_alg->completeEdge(i);
	    }
#else
        printf("PAG with PES optimization disabled.\n");
#endif
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < vecsize; i++) {
            appr_alg->expandSpace(i);
	    }  
#pragma omp parallel for schedule(dynamic)		
        for (int i = 0; i < vecsize; i++) {
            appr_alg->addEdgeProj(i);
	    }  		

        appr_alg->findCenterNeighbor(center_data.data());

	    appr_alg->compression(vecsize);
		cout << "FastGraph build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";    		
        appr_alg->saveIndex();
        input.close();	
    }
    return;
}
