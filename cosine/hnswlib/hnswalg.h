#pragma once

#include "visited_list_pool.h"
#include "paglib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <cmath>
#include <list>
#include <cstring>
#include <immintrin.h> 

inline void write_low4(unsigned char* dst, unsigned char val) {
    *dst = (*dst & 0xF0) | (val & 0x0F);
}

inline void write_high4(unsigned char* dst, unsigned char val) {
    *dst = (*dst & 0x0F) | ((val & 0x0F) << 4);
}

inline unsigned char get_low4(unsigned char val) {
    return val & 0x0F;
}

inline unsigned char get_high4(unsigned char val) {
    return (val >> 4) & 0x0F;
}

inline __m512i expand_4bit_to_zmm_i(uint64_t lvl64) {
    __m128i lvl = _mm_cvtsi64_si128(lvl64);
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m128i idx_lo = _mm_and_si128(lvl, mask_lo);
    __m128i idx_hi = _mm_and_si128(_mm_srli_epi64(lvl, 4), mask_lo);
    __m128i result_128 = _mm_unpacklo_epi8(idx_hi, idx_lo);
    return _mm512_cvtepu8_epi32(result_128);
}

static inline int InsertIntoPool(Neighbor* addr, unsigned K, const Neighbor& nn) {
    float d = nn.distance;

    if (d < addr[0].distance) {
        memmove(addr + 1, addr, K * sizeof(Neighbor));
        addr[0] = nn;
        return 0;
    }

    unsigned left = 0, right = K - 1;
    while (right - left > 1) {
        unsigned mid = (left + right) >> 1;
        if (addr[mid].distance > d)
            right = mid;
        else
            left = mid;
    }
	
    memmove(addr + right + 1, addr + right, (K - right) * sizeof(Neighbor));
    
    addr[right] = nn;

    return right;
}

static inline int InsertIntoPoolIndex(NeighborIndex* addr, unsigned K, const NeighborIndex& nn)
{
    float d = nn.distance;

    unsigned idx = K; 
    unsigned left = 0;
    unsigned right = K; 

    while (left < right) {
        unsigned mid = left + (right - left) / 2;
        if (addr[mid].distance < d) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    idx = left; 

    size_t num_elements_to_move = (K - 1) - idx;
    
    if (num_elements_to_move > 0) {
        memmove(addr + idx + 1, addr + idx, num_elements_to_move * sizeof(NeighborIndex));
    }
    
    addr[idx] = nn;

    return idx;
}

static inline int InsertIntoPool2(Neighbor* addr, unsigned K, const Neighbor& nn)
{
    float d = nn.distance;

    unsigned idx = K; 
    unsigned left = 0;
    unsigned right = K; 

    while (left < right) {
        unsigned mid = left + (right - left) / 2;
        if (addr[mid].distance < d) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    idx = left; 

    size_t num_elements_to_move = (K - 1) - idx;
    
    if (num_elements_to_move > 0) {
        memmove(addr + idx + 1, addr + idx, num_elements_to_move * sizeof(Neighbor));
    }
    
    addr[idx] = nn;

    return idx;
}

struct Neighbor2 {
    unsigned int id;
    float distance;
};

const size_t MAX_NEIGHBORS = 64;

void insert_neighbor(std::vector<std::vector<Neighbor2>> &addNeighbors0,
                     size_t index, unsigned int id, float distance) {
						 
    Neighbor2 newNeighbor{id, distance};
    auto &bucket = addNeighbors0[index];
    auto it = std::lower_bound(bucket.begin(), bucket.end(), newNeighbor,
                               [](const Neighbor2 &a, const Neighbor2 &b) {
                                   return a.distance < b.distance;
                               });
    bucket.insert(it, newNeighbor);
    if (bucket.size() > MAX_NEIGHBORS) {
        bucket.pop_back();
    }
}

void delete_neighbor(std::vector<std::vector<Neighbor2>> &addNeighbors0,
                     size_t index, unsigned int id) {
    if (index >= addNeighbors0.size()) return; 

    auto &bucket = addNeighbors0[index];
    auto it = std::find_if(bucket.begin(), bucket.end(),
                           [id](const Neighbor2 &n) { return n.id == id; });

    if (it != bucket.end()) {
        bucket.erase(it);
    }
}

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t> *s) {
        }
        
        HierarchicalNSW(SpaceInterface<dist_t> *s1, SpaceInterface<dist_t> *s2, SpaceInterface<dist_t> *s3, const char* path_index, bool nmslib = false, size_t max_elements=0) {
			loadIndex(path_index, s1, s2, s3, max_elements);		
        }

        HierarchicalNSW(size_t maxk, int step, size_t m, size_t vecdim, std::vector<std::vector<std::vector<float>>>& projVec0, int level, int subdim, std::vector<int>& permutation, SpaceInterface<dist_t> *s1, SpaceInterface<dist_t> *s2, SpaceInterface<dist_t> *s3, size_t max_elements, char* path_index, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
                link_list_locks_(max_elements), element_levels_(max_elements) {
            
			max_elements_ = max_elements;

            maxk_ = maxk;
            step_ = step;
			
            permutation_ = permutation;
            vecsize_ = max_elements;
			projVec = projVec0;
			//normInfo = norm;
			vecdim_ = vecdim;

	        padded_dim_ = (vecdim_ + 15) & ~0xF;
            extended_dim_ = ((padded_dim_ + level - 1) / level) * level;

            fstdistfunc_ = s1->get_dist_func();
			fstipfunc_ = s2->get_dist_func();		
            dist_func_param_ = s1->get_dist_func_param();	
			fstsubfunc_ = s3->get_dist_func();
			sub_func_param_ = s3->get_dist_func_param();
			
            level_ = level;
            subdim_ = subdim;

            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;
			m_ = m;
			
			int half_level = level_ / 2;
			
			offset0 = 0;
            offset1 = 16 * (half_level);
			offset2 = 16 * (half_level + sizeof(int16_t));

			segment_size_ = 16 * (half_level + 2 * sizeof(int16_t));
            segment_size_new_ = 16 * (half_level + 2 * sizeof(int16_t));

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            //pre_size_links_level0_ = maxM0_ * (3 * sizeof(float) + sizeof(tableint) + half_level);
			size_links_level0_ = maxM0_ * (2 * sizeof(int16_t) + sizeof(tableint) + half_level) + sizeof(linklistsizeint);   //new
            
			data_size_ = padded_dim_ * sizeof(int16_t);  //int8
			
			size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype) + 2 * sizeof(float); 
            offsetData_ = size_links_level0_;
            label_offset_ = data_size_ + sizeof(float);
            sec_part_ = data_size_ + sizeof(float) + sizeof(labeltype);
            
            init_size_ = maxM0_ * sizeof(tableint) + segment_size_ + sizeof(linklistsizeint);
			data_level0_memory_ = (char **) malloc(vecsize_*sizeof(char*));
			for(int i = 0; i < vecsize_; i++){
				data_level0_memory_[i] = (char *) malloc(init_size_);
			}
			
			vec_level0_memory_ = (char *) malloc(vecsize_ * sec_part_);
			
            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            enterpoint_node_ = -1;
            maxlevel_ = -1;

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
			addNeighbors.resize(vecsize_);
			is_full = new bool[vecsize_]();
			start_pos = new int[vecsize_];
			
            std::string folderPath(path_index);
            std::string fullPath;
            if (!folderPath.empty() && (folderPath.back() == '/' || folderPath.back() == '\\')) {
                indexPath_ = folderPath + "index.bin";
                infoPath_ = folderPath + "info.bin";				
            } else {
                indexPath_ = folderPath + "/index.bin";
                infoPath_ = folderPath + "/info.bin";				
            }				
        }

        inline float dot_product_avx512(const float* __restrict a, const float* __restrict b) const{
    
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();
    
            int i = 0;
            int limit = padded_dim_ & (~63); 
    
            for (; i < limit; i += 64) {
                __m512 va0 = _mm512_loadu_ps(a + i);
                __m512 vb0 = _mm512_loadu_ps(b + i);
                sum0 = _mm512_fmadd_ps(va0, vb0, sum0);

                __m512 va1 = _mm512_loadu_ps(a + i + 16);
                __m512 vb1 = _mm512_loadu_ps(b + i + 16);
                sum1 = _mm512_fmadd_ps(va1, vb1, sum1);

               __m512 va2 = _mm512_loadu_ps(a + i + 32);
               __m512 vb2 = _mm512_loadu_ps(b + i + 32);
               sum2 = _mm512_fmadd_ps(va2, vb2, sum2);

               __m512 va3 = _mm512_loadu_ps(a + i + 48);
               __m512 vb3 = _mm512_loadu_ps(b + i + 48);
               sum3 = _mm512_fmadd_ps(va3, vb3, sum3);
            }

            __m512 final_sum = _mm512_add_ps(sum0, sum1);
            final_sum = _mm512_add_ps(final_sum, sum2);
            final_sum = _mm512_add_ps(final_sum, sum3);

            float result = _mm512_reduce_add_ps(final_sum);

            for (; i < padded_dim_; i++) {
                result += a[i] * b[i]; 
            }
            return result;
        }

        float find_max_abs(const float* __restrict vec) const{
            __m512 max_abs_vec = _mm512_setzero_ps(); 

            for (int i = 0; i < padded_dim_; i += 16) {
                __m512 v = _mm512_loadu_ps(vec + i);
        
                __m512 abs_v = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), v);

                max_abs_vec = _mm512_max_ps(max_abs_vec, abs_v);
            }

            float max_abs = _mm512_reduce_max_ps(max_abs_vec);
            return std::max(max_abs, 1e-6f);
        }

        inline int16_t float_to_int16(float x, float scale) const {
            float scaled = x * scale;
    
            int32_t q_int32 = static_cast<int32_t>(std::round(scaled));
    
            int16_t q = static_cast<int16_t>(q_int32);

            const int16_t MAX_VAL = 32767;
            const int16_t MIN_VAL = -32767;
    
            if (q_int32 > MAX_VAL) {
                q = MAX_VAL;
            } else if (q_int32 < MIN_VAL) {
                q = MIN_VAL;
            }    
                return q;
        }

        inline float dot_product_avx512_f32_i16(  //warning
            const float* __restrict q,      // float32 query
            const int16_t* __restrict vq,   // int16 quantized vector
            float scale_v                  // quantization scale
        )const
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            const __m512 inv_scale = _mm512_set1_ps(1.0f / scale_v);

            int i = 0;
            int limit = padded_dim_ & ~31; // multiples of 32

            for (; i < limit; i += 32) {
                __m512 q0 = _mm512_loadu_ps(q + i);
                __m512 q1 = _mm512_loadu_ps(q + i + 16);

                // ---- v0 ----
                __m256i v0_i16 = _mm256_loadu_si256((const __m256i*)(vq + i));
                __m512i v0_i32 = _mm512_cvtepi16_epi32(v0_i16);
                __m512  v0     = _mm512_cvtepi32_ps(v0_i32);

                // ---- v1 ----
                __m256i v1_i16 = _mm256_loadu_si256((const __m256i*)(vq + i + 16));
                __m512i v1_i32 = _mm512_cvtepi16_epi32(v1_i16);
                __m512  v1     = _mm512_cvtepi32_ps(v1_i32);

                v0 = _mm512_mul_ps(v0, inv_scale);
                v1 = _mm512_mul_ps(v1, inv_scale);

                acc0 = _mm512_fmadd_ps(q0, v0, acc0);
                acc1 = _mm512_fmadd_ps(q1, v1, acc1);
            }

            if (i < padded_dim_) {
                __m512 q_tail = _mm512_loadu_ps(q + i);

                __m256i v_tail_i16 = _mm256_loadu_si256((const __m256i*)(vq + i));
                __m512i v_tail_i32 = _mm512_cvtepi16_epi32(v_tail_i16);
                __m512  v_tail     = _mm512_cvtepi32_ps(v_tail_i32);

                v_tail = _mm512_mul_ps(v_tail, inv_scale);
                acc0   = _mm512_fmadd_ps(q_tail, v_tail, acc0);
            }

            __m512 final_acc = _mm512_add_ps(acc0, acc1);
            float dot = _mm512_reduce_add_ps(final_acc);

            return dot;
        }

        inline float dot_product_avx512_int16(
            const int16_t* __restrict qa,
            const int16_t* __restrict qb,
            float scale_a,
            float scale_b
        ) const
        {
            __m512i acc64 = _mm512_setzero_si512();
            const __m512i zero = _mm512_setzero_si512();

            int i = 0;

            int limit = padded_dim_ & ~31;

            for (; i < limit; i += 32) {
                __m512i va = _mm512_loadu_si512((const __m512i*)(qa + i));
                __m512i vb = _mm512_loadu_si512((const __m512i*)(qb + i));

                __m512i prod32 = _mm512_dpwssd_epi32(zero, va, vb);

                acc64 = _mm512_add_epi64(acc64, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(prod32, 0)));
                acc64 = _mm512_add_epi64(acc64, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(prod32, 1)));
            }

            int64_t dot = _mm512_reduce_add_epi64(acc64);

            if (i < padded_dim_) {
                __m256i va = _mm256_loadu_si256((const __m256i*)(qa + i));
                __m256i vb = _mm256_loadu_si256((const __m256i*)(qb + i));

                __m256i prod32 = _mm256_madd_epi16(va, vb);

                alignas(32) int32_t tmp[8];
                _mm256_store_si256((__m256i*)tmp, prod32);
                for (int k = 0; k < 8; ++k)
                    dot += tmp[k];
            }

            return static_cast<float>(dot) / (scale_a * scale_b);
        }

        inline float dot_product_avx512_extended(const float* __restrict a, const float* __restrict b) const {
        
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();
        
            int i = 0;
            int limit = extended_dim_ & (~63); 

            for (; i < limit; i += 64) {
                __m512 va0 = _mm512_loadu_ps(a + i);
                __m512 vb0 = _mm512_loadu_ps(b + i);
                sum0 = _mm512_fmadd_ps(va0, vb0, sum0); // FMA: va0*vb0 + sum0

                __m512 va1 = _mm512_loadu_ps(a + i + 16);
                __m512 vb1 = _mm512_loadu_ps(b + i + 16);
                sum1 = _mm512_fmadd_ps(va1, vb1, sum1);

                __m512 va2 = _mm512_loadu_ps(a + i + 32);
                __m512 vb2 = _mm512_loadu_ps(b + i + 32);
                sum2 = _mm512_fmadd_ps(va2, vb2, sum2);

                __m512 va3 = _mm512_loadu_ps(a + i + 48);
                __m512 vb3 = _mm512_loadu_ps(b + i + 48);
                sum3 = _mm512_fmadd_ps(va3, vb3, sum3);
            }

            __m512 final_sum = _mm512_add_ps(sum0, sum1);
            final_sum = _mm512_add_ps(final_sum, sum2);
            final_sum = _mm512_add_ps(final_sum, sum3);

            float result = _mm512_reduce_add_ps(final_sum);

            for (; i < extended_dim_; i++) {
                result += a[i] * b[i]; 
            }
    
            return result;
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW() {
            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
			if(projVecQuery != NULL){
				delete[] projVecQuery;
			}
        }

	    std::string	indexPath_;
	    std::string infoPath_;
        
		int* start_pos;
		bool* is_full;
		size_t init_size_;
		char* init_vec_;
		size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;
		
		size_t maxk_;
        int step_;

        int level_;
        int subdim_;	

        //std::vector<float> normInfo;  
        std::vector<std::vector<std::vector<float>>> projVec;
		
		float* projVecQuery;
        size_t vecdim_;
	    size_t padded_dim_;
        size_t extended_dim_;		
		
		size_t vecsize_;
		size_t total_size;
        size_t index_size_new;	    

        size_t segment_size_;
        size_t segment_size_new_;		
		
        size_t M_;
        size_t m_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;
		int* inverse_id;

        double mult_, revSize_;
        int maxlevel_;

        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;
        std::vector<int> permutation_;	
        std::vector<std::mutex> link_list_locks_;
        tableint enterpoint_node_;

        size_t size_links_level0_;
        size_t pre_size_links_level0_;		
        size_t offsetData_;

		size_t sec_part_;
		
        size_t* size_data_per_element_new_;
        size_t* size_links_offset_;
        tableint* num_edge_group;
        tableint* num_edge_offset;		

        std::vector<int> center_neighbor;

		size_t offset0;
        size_t offset1;
		size_t offset2;

        char **data_level0_memory_;
		char *data_level0_memory_query_;
		char *vec_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;

        std::vector<std::vector<Neighbor2>> addNeighbors;

        size_t data_size_;
        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
		DISTFUNC<dist_t> fstipfunc_;
		DISTFUNC<dist_t> fstsubfunc_;
		
        void *dist_func_param_;
		void *sub_func_param_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            return (labeltype *) (vec_level0_memory_ + internal_id * sec_part_ + label_offset_);
        }

        inline unsigned char* getQuanVal(char* data, int i, int j) const { //j
			int a = i / 16;
			int b = i % 16; 			
			char* data_offset = data + a * segment_size_ + 8 * j + ((int)(b/2));
			return (unsigned char* )data_offset;
        }

        inline uint16_t* getTmpVal(char* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data_offset = data + a * segment_size_ + offset1 + b * sizeof(int16_t);
			return (uint16_t* )data_offset;
        }

        inline uint16_t* getTmpVal2(char* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data_offset = data + a * segment_size_ + offset2 + b * sizeof(int16_t);
			return (uint16_t* )data_offset;
        }

        inline char *getNormByInternalIdQuery(tableint internal_id) const {
            return (vec_level0_memory_ + internal_id * sec_part_);
        }

        inline char *getNormByInternalId(tableint internal_id) const {
            return (vec_level0_memory_ + internal_id * sec_part_);
        }
		
        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;

			//float qnorm = fstipfunc_(data_point, data_point, dist_func_param_);
			float* norm_pointer = (float*)getNormByInternalId(ep_id);
			//float v_norm = *norm_pointer;
			//norm_pointer += 1;
			float v_scale = *norm_pointer;
			norm_pointer += 1;			
			dist_t dist = 2 - 2*dot_product_avx512_f32_i16((float*)data_point, (int16_t*)norm_pointer, v_scale);

			top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data = (int*)get_linklist(curNodeNum, layer);
                size_t size = getListCount((linklistsizeint*)data);
                int *datal = data + 1;
				
				int* data2 = datal;
				int* data3 = datal + 1;
				
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data2)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data2) + 64), _MM_HINT_T0);
                _mm_prefetch(getNormByInternalId(*data2), _MM_HINT_T0);
                _mm_prefetch(getNormByInternalId(*(data3)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
					int candidate_id = *(datal+j);
					int* data4 = datal + j;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data4)), _MM_HINT_T0);
                    _mm_prefetch(getNormByInternalId(*(data4)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
 
			        norm_pointer = (float*)getNormByInternalId(candidate_id);
			        //v_norm = *norm_pointer;
			        //norm_pointer += 1;
			        v_scale = *norm_pointer;
			        norm_pointer += 1;			
			        float dist1 = 2 - 2*dot_product_avx512_f32_i16((float*)data_point, (int16_t*)norm_pointer, v_scale);

                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getNormByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif
                        top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerFast(tableint cur_id, tableint ep_id, const void *query_point){
			std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

            float* query_extended_point = new float[extended_dim_];
			permute(query_extended_point, (float* ) query_point);

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

			int LL = ef_construction_;
            std::vector<Neighbor> retset(LL + 1);
            
            const int TOTAL_ELEMENTS = level_ * 2 * m_; 
            const int ALIGNMENT = 64; 

            float* table = (float*) std::aligned_alloc(ALIGNMENT, TOTAL_ELEMENTS * sizeof(float));
            std::memset(table, 0, TOTAL_ELEMENTS * sizeof(float)); 

            for(int j = 0; j < level_; j++){
                float* y = query_extended_point + j * subdim_;
                float* current_level_ptr = table + j * (2 * m_);
                for(int i = 0; i < m_; i++){  
                    _mm_prefetch(reinterpret_cast<const char*>(projVec[j][i].data()), _MM_HINT_T0);
                    current_level_ptr[i] = -1 * fstsubfunc_((void*) y, (void*) projVec[j][i].data(), sub_func_param_);
                    current_level_ptr[i + m_] = -1 * current_level_ptr[i];
                }
            }
		
			//float qnorm = fstipfunc_(query_point, query_point, dist_func_param_)/2;		
			float* norm_pointer0 = (float*)getNormByInternalId(ep_id);
			//float v_norm0 = *norm_pointer0;
			//norm_pointer0 += 1;
			float v_scale0 = *norm_pointer0;
			norm_pointer0 += 1;	
            float ip0 = dot_product_avx512_f32_i16((float*)query_point, (int16_t*)norm_pointer0, v_scale0);
            float dist0 = 0.5 - ip0;        

            retset[0] = Neighbor(ep_id, dist0, ip0, true);
            visited_array[ep_id] = visited_array_tag; 

            int k = 0;
			int l_num = 1;
			
            float PORTABLE_ALIGN64 Thres[16];				
			int real_data[maxM0_];

            const __m128i m128_4 = _mm_set1_epi8(0x0F);
            while (k < l_num) {
                int nk = l_num;

                if (retset[k].flag) {
                    retset[k].flag = false;
                    unsigned n = retset[k].id;
                    float cur_dist = retset[k].distance;

                    std::unique_lock <std::mutex> lock(link_list_locks_[n]);

					if(l_num < LL){	
                        int *data; 
						data = (int *) get_linklist0(n);				
                        size_t size = getListCount((linklistsizeint*)data);
						int* datal = data + 1;
				        int* data2 = datal;
						int* data3 = datal + 1;		

                        char* tmp_datal = (char*) datal;
                        //float* cur_norm = (float *) (tmp_datal + pre_size_links_level0_);

#ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *data2), _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + *data2 + 64), _MM_HINT_T0);
                        _mm_prefetch(getNormByInternalId(*data2), _MM_HINT_T0);
                        _mm_prefetch((char *) (data3), _MM_HINT_T0);
                        //_mm_prefetch((char *) cur_norm, _MM_HINT_T0);						
#endif

                        bool reverse_flag = true;
                        for (size_t j = 1; j <= size; j++) {					
							int candidate_id = *(datal + j - 1);
							int* data4 = datal + j;
							             
#ifdef USE_SSE
                            _mm_prefetch((char *) (visited_array + *data4), _MM_HINT_T0);
                            _mm_prefetch(getNormByInternalId(*data4), _MM_HINT_T0);
#endif

                            if (!(visited_array[candidate_id] == visited_array_tag)) {
                        
                                visited_array[candidate_id] = visited_array_tag;
	
			                    float* norm_pointer = (float*)getNormByInternalId(candidate_id);
			                    //float v_norm = *norm_pointer;
			                    //norm_pointer += 1;
			                    float v_scale = *norm_pointer;
			                    norm_pointer += 1;	
                                float ip_ = dot_product_avx512_f32_i16((float*)query_point, (int16_t*)norm_pointer, v_scale);
                                float dist = 0.5 - ip_;        
	
                                if(reverse_flag == true && dist < cur_dist){
									reverse_flag = false;
								}
						
                                if (l_num == LL && dist >= retset[LL - 1].distance ) continue;

                                int r;
		                        if(l_num == LL){
                 		            Neighbor nn2(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), LL, nn2);
		                        }
	                 	        else {
                                    Neighbor nn(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), l_num, nn);
			                        l_num++;
                                }
                                if (r < nk) {nk = r;}
						    }							
					    }
						
						if(reverse_flag == true){
							insert_neighbor(addNeighbors, n, cur_id, (cur_dist + 0.5)*2);
						}
					}
					else{
						//float difference_ = cur_dist - retset[LL - 1].distance;
						
						//float lowerBound = retset[LL - 1].distance;
						int* data;
                        data = (int *) get_linklist0(n);						
						int* datal = data + 1;

                        char* tmp_datal = (char*) datal;						

						size_t size = getListCount((linklistsizeint*)data);
						
						float lowerBound = retset[LL-1].distance; 
						float qcosine = retset[k].ip;
						float Lbound = qcosine;

                        int count = 0;            

						int div = size % 16;
						int round;
						if(div == 0){
							round = size / 16;
							div = 16;
						}else{
							round = size / 16 + 1;
						}							

						//__m512 v_val = _mm512_set1_ps(val); //warning
						__m512 v_qcosine = _mm512_set1_ps(qcosine);
                        
						bool reverse_flag = true;					
					    
						unsigned char* char_pointer = (unsigned char*) (datal + maxM0_);
                        for(int rr = 0; rr < round; rr++){
							int round2 = level_ / 8;
							__m512 sum = _mm512_setzero_ps();

for (int rr2 = 0; rr2 < round2; rr2++) {
    __m512i all_levels = _mm512_loadu_si512((const __m512i*)char_pointer);
    const float* t_ptr = table + rr2 * 128;

    __m128i r0 = _mm512_extracti32x4_epi32(all_levels, 0);
    __m128i r0_lo = _mm_and_si128(r0, m128_4);
    __m128i r0_hi = _mm_and_si128(_mm_srli_epi64(r0, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r0_hi, r0_lo)), _mm512_load_ps(t_ptr + 0)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r0_hi, r0_lo)), _mm512_load_ps(t_ptr + 16)));

    __m128i r1 = _mm512_extracti32x4_epi32(all_levels, 1);
    __m128i r1_lo = _mm_and_si128(r1, m128_4);
    __m128i r1_hi = _mm_and_si128(_mm_srli_epi64(r1, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r1_hi, r1_lo)), _mm512_load_ps(t_ptr + 32)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r1_hi, r1_lo)), _mm512_load_ps(t_ptr + 48)));

    __m128i r2 = _mm512_extracti32x4_epi32(all_levels, 2);
    __m128i r2_lo = _mm_and_si128(r2, m128_4);
    __m128i r2_hi = _mm_and_si128(_mm_srli_epi64(r2, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r2_hi, r2_lo)), _mm512_load_ps(t_ptr + 64)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r2_hi, r2_lo)), _mm512_load_ps(t_ptr + 80)));

    __m128i r3 = _mm512_extracti32x4_epi32(all_levels, 3);
    __m128i r3_lo = _mm_and_si128(r3, m128_4);
    __m128i r3_hi = _mm_and_si128(_mm_srli_epi64(r3, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r3_hi, r3_lo)), _mm512_load_ps(t_ptr + 96)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r3_hi, r3_lo)), _mm512_load_ps(t_ptr + 112)));

    char_pointer += 64;
}
							
uint16_t* bf16_ptr = (uint16_t*)char_pointer;

__m512i bf16_01 = _mm512_loadu_si512((const __m512i*)bf16_ptr);
bf16_ptr += 32;

__m256i bf16_0 = _mm512_castsi512_si256(bf16_01);                 // low 16
__m256i bf16_1 = _mm512_extracti64x4_epi64(bf16_01, 1);           // high 16

__m512i v0_i32 = _mm512_cvtepu16_epi32(bf16_0);
v0_i32 = _mm512_slli_epi32(v0_i32, 16);
__m512 v_float = _mm512_castsi512_ps(v0_i32);

sum = _mm512_fmadd_ps(v_float, v_qcosine, sum);

__m512i v1_i32 = _mm512_cvtepu16_epi32(bf16_1);
v1_i32 = _mm512_slli_epi32(v1_i32, 16);
__m512 v_mul = _mm512_castsi512_ps(v1_i32);

sum = _mm512_fmadd_ps(sum, v_mul, _mm512_set1_ps(0.5));
_mm512_store_ps(Thres, sum);
	
							int check_num = 16;
							if(rr == round - 1) check_num = div;
							
						    for(int i = 0; i < check_num; i++){
							    if(Thres[i] <= lowerBound){
                                    real_data[count] = datal[i];
								    count++;
							        if(reverse_flag == true && Thres[i] <= cur_dist)
								        reverse_flag = false;															
							    }
						    }
							char_pointer = (unsigned char*) bf16_ptr;
                            datal += 16;
                            //cur_norm += 16;							
						}
						
						if(reverse_flag == true){
						    insert_neighbor(addNeighbors, n, cur_id, 2.0 * (cur_dist + 0.5));
					    }

                        size = count;
						datal = real_data;

						int* data2 = datal;
						int* data3 = datal+1;

#ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *data2), _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + *data2 + 64), _MM_HINT_T0);
                        _mm_prefetch(getNormByInternalId(*data2), _MM_HINT_T0);
                        _mm_prefetch((char *) (data3), _MM_HINT_T0);
#endif
                        for (size_t j = 0; j < size; j++) {
							int candidate_id = *(datal+j);
							int* data4 = datal+j+1;

#ifdef USE_SSE
						    if(j < size-1){
                                _mm_prefetch((char *) (visited_array + *data4), _MM_HINT_T0);
                                _mm_prefetch(getNormByInternalId(*data4), _MM_HINT_T0);
							}
#endif							       
                            if (!(visited_array[candidate_id] == visited_array_tag)) {
								visited_array[candidate_id] = visited_array_tag;

			                    float* norm_pointer = (float*)getNormByInternalId(candidate_id);
			                    //float v_norm = *norm_pointer;
			                    //norm_pointer += 1;
			                    float v_scale = *norm_pointer;
			                    norm_pointer += 1;	
                                float ip_ = dot_product_avx512_f32_i16((float*)query_point, (int16_t*)norm_pointer, v_scale);
                                float dist = 0.5 - ip_;  

                                if (l_num == LL && dist >= retset[LL - 1].distance ) continue;

                                int r;
		                        if(l_num == LL){
 			
                 		            Neighbor nn2(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), LL, nn2);
		                        }
	                 	        else {
                                    Neighbor nn(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), l_num, nn);
			                        l_num++;									
                                }
                                if (r < nk) {nk = r;}						
							}												
					    }				
					} 					
				}
				if (nk <= k)
                k = nk;
                else {++k;}
			}			
				
            visited_list_pool_->releaseVisitedList(vl);
			for(int i = 0; i < l_num; i++){
			    top_candidates.emplace( (retset[i].distance + 0.5)*2, retset[i].id);
			}
			delete[] query_extended_point;
            std::free(table);			
			return top_candidates;				
		} 

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerFast2(tableint cur_id, const void *query_point){

            int step = step_;
					
			std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

			//float qnorm = 0.5;		

            //float true_qnorm = 1.0f;

			//float qnorm_vec[padded_dim_];
			float* cur_query_point = (float*) query_point;
			//for(int i = 0; i < padded_dim_; i++)
			//	qnorm_vec[i] = cur_query_point[i];
			
			float q_scale = 32767.0f / find_max_abs(cur_query_point);
			
			int16_t query_point_int16[padded_dim_];
			for(int i = 0; i < padded_dim_; i++)
				query_point_int16[i] = float_to_int16(cur_query_point[i], q_scale);	
			
            float* query_extended_point = new float[extended_dim_];
			permute(query_extended_point, cur_query_point);

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
			
			int max_round2 = ef_construction_ / step;  //change 			

            int LL = step;
			int LL2 = step;
			int LL3 = step;
		
            std::vector<NeighborIndex> retset(LL);
			std::vector<NeighborIndex> retset2(LL2); 
			std::vector<NeighborIndex> retset3(LL3);			
					
			int head2 = 0;
			int head3 = 0;
			int size2 = 0;
			int size3 = 0;
	
			//float* norm_pointer0 = (float*)getNormByInternalId(ep_id);
			//float v_norm0 = *norm_pointer0;
			//norm_pointer0 += 1;
			//float v_scale0 = *norm_pointer0;
			//norm_pointer0 += 1;	
            //float ip0 = dot_product_avx512_int16(query_point_int16, (int16_t*)norm_pointer0, q_scale, v_scale0);
            //float dist0 =  0.5 - ip0;     

            //retset[0] = NeighborIndex(ep_id, dist0, ip0, true);
            //visited_array[ep_id] = visited_array_tag; 

            //int cur_size = 1;
			for(int i = 0; i < step; i++){
                //if (cur_size >= step) {
                //    break; 
                //}				
				
				//if(ep_id == i) continue;

			    float* norm_pointer = (float*)getNormByInternalId(i);
			    //float v_norm = *norm_pointer;
			    //norm_pointer += 1;
			    float v_scale = *norm_pointer;
			    norm_pointer += 1;	
                float ip = dot_product_avx512_int16(query_point_int16, (int16_t*)norm_pointer, q_scale, v_scale);
                float dist = 0.5 - ip;     

                retset[i] = NeighborIndex(i, dist, ip, true);
				//cur_size++;
                visited_array[i] = visited_array_tag;			
			}
					
            std::sort(retset.begin(), retset.begin() + step);
			
            const int TOTAL_ELEMENTS = level_ * 2 * m_; 
            const int ALIGNMENT = 64; 

            float* table = (float*) std::aligned_alloc(ALIGNMENT, TOTAL_ELEMENTS * sizeof(float));
            std::memset(table, 0, TOTAL_ELEMENTS * sizeof(float)); 

            for(int j = 0; j < level_; j++){
                float* y = query_extended_point + j * subdim_;
                float* current_level_ptr = table + j * (2 * m_);
                for(int i = 0; i < m_; i++){  
                    _mm_prefetch(reinterpret_cast<const char*>(projVec[j][i].data()), _MM_HINT_T0);
                    current_level_ptr[i] = -1 * fstsubfunc_((void*) y, (void*) projVec[j][i].data(), sub_func_param_);
                    current_level_ptr[i + m_] = -1 * current_level_ptr[i];
                }
            }
		
		    int retset_size = step;

            float PORTABLE_ALIGN64 Thres[16];				
			int real_data[maxM0_];


            const __m128i m128_4 = _mm_set1_epi8(0x0F);
			for(int seg = 0; seg < max_round2; seg++){			
				if(retset_size < 1) break;
					
                int k = 0;
			    int next_k = 0;	
                int next_next_k = 0;			

                while (k < retset_size) {
                    retset[k].flag = false;
                    float cur_dist = retset[k].distance;
					unsigned n = retset[k].id;
					float LowerBound = retset[retset_size-1].distance;

			        if(k == next_k){
					    next_k = retset_size;
					    for(int ii = k+1; ii < retset_size; ii++){
						    if(retset[ii].flag == true){
							    next_k = ii;
							    _mm_prefetch((char *) (get_linklist0(retset[ii].id)), _MM_HINT_T0);
							     break;
						    }
					    }
					}

					std::unique_lock <std::mutex> lock(link_list_locks_[n]);

					int* data = (int*)get_linklist0(n);							
					int size = *data;
					
					int* datal = data + 1;
					//float* cur_norm = (float *) (datal + pre_size_links_level0_);						
						 
                    int count = 0;            

					int div = size % 16;
					int round;
					if(div == 0){
						round = size / 16;
						div = 16;
					}else{
						round = size / 16 + 1;
					}	

					__m512 temp_center = _mm512_set1_ps(retset[k].ip);
                    unsigned char* char_pointer = (unsigned char*) (datal + maxM0_);

				    bool reverse_flag = true;							
					for(int rr = 0; rr < round; rr++){
						int round2 = level_ / 8;
						__m512 sum = _mm512_setzero_ps();
							
for (int rr2 = 0; rr2 < round2; rr2++) {
    __m512i all_levels = _mm512_loadu_si512((const __m512i*)char_pointer);
    const float* t_ptr = table + rr2 * 128;

    __m128i r0 = _mm512_extracti32x4_epi32(all_levels, 0);
    __m128i r0_lo = _mm_and_si128(r0, m128_4);
    __m128i r0_hi = _mm_and_si128(_mm_srli_epi64(r0, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r0_hi, r0_lo)), _mm512_load_ps(t_ptr + 0)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r0_hi, r0_lo)), _mm512_load_ps(t_ptr + 16)));

    __m128i r1 = _mm512_extracti32x4_epi32(all_levels, 1);
    __m128i r1_lo = _mm_and_si128(r1, m128_4);
    __m128i r1_hi = _mm_and_si128(_mm_srli_epi64(r1, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r1_hi, r1_lo)), _mm512_load_ps(t_ptr + 32)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r1_hi, r1_lo)), _mm512_load_ps(t_ptr + 48)));

    __m128i r2 = _mm512_extracti32x4_epi32(all_levels, 2);
    __m128i r2_lo = _mm_and_si128(r2, m128_4);
    __m128i r2_hi = _mm_and_si128(_mm_srli_epi64(r2, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r2_hi, r2_lo)), _mm512_load_ps(t_ptr + 64)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r2_hi, r2_lo)), _mm512_load_ps(t_ptr + 80)));

    __m128i r3 = _mm512_extracti32x4_epi32(all_levels, 3);
    __m128i r3_lo = _mm_and_si128(r3, m128_4);
    __m128i r3_hi = _mm_and_si128(_mm_srli_epi64(r3, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r3_hi, r3_lo)), _mm512_load_ps(t_ptr + 96)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r3_hi, r3_lo)), _mm512_load_ps(t_ptr + 112)));

    char_pointer += 64;
}
						
uint16_t* bf16_ptr = (uint16_t*)char_pointer;

__m512i bf16_01 = _mm512_loadu_si512((const __m512i*)bf16_ptr);
bf16_ptr += 32;

__m256i bf16_0 = _mm512_castsi512_si256(bf16_01);                 // low 16
__m256i bf16_1 = _mm512_extracti64x4_epi64(bf16_01, 1);           // high 16

__m512i v0_i32 = _mm512_cvtepu16_epi32(bf16_0);
v0_i32 = _mm512_slli_epi32(v0_i32, 16);
__m512 v_float = _mm512_castsi512_ps(v0_i32);

sum = _mm512_fmadd_ps(v_float, temp_center, sum);

__m512i v1_i32 = _mm512_cvtepu16_epi32(bf16_1);
v1_i32 = _mm512_slli_epi32(v1_i32, 16);
__m512 v_mul = _mm512_castsi512_ps(v1_i32);

//__m256i bf16_2 = _mm256_loadu_si256((const __m256i*)bf16_ptr);
//bf16_ptr += 16;

//__m512i v2_i32 = _mm512_cvtepu16_epi32(bf16_2);
//v2_i32 = _mm512_slli_epi32(v2_i32, 16);
//__m512 v_add = _mm512_castsi512_ps(v2_i32);

sum = _mm512_fmadd_ps(sum, v_mul, _mm512_set1_ps(0.5));
_mm512_store_ps(Thres, sum);
											
 						int check_num = 16;
						if(rr == round - 1) check_num = div;
							
						for(int i = 0; i < check_num; i++){
							if(Thres[i] <= LowerBound){
                                real_data[count] = datal[i];
								count++;
							    if(reverse_flag == true && Thres[i] <= cur_dist)
								    reverse_flag = false;															
							}
						}
 
                        char_pointer = (unsigned char*) bf16_ptr;							
                        datal += 16;	
					}

					if(count == 0){
					    k = next_k;
					    continue;						
					}

					if(reverse_flag == true){
						insert_neighbor(addNeighbors, n, cur_id, 2.0 * (cur_dist + 0.5));
					}

                    next_next_k = next_k;
                 
                    size = count;			
				    datal = real_data;
					
                    for (size_t j = 0; j < size; j++) {
					    int candidate_id = datal[j];
					    int* data2; 
						
						if(j < size-1){
						    data2 = datal+j+1;
						//int* data3 = datal+j+2;

                        _mm_prefetch((char *) (visited_array + *data2), _MM_HINT_T0);
                        _mm_prefetch(getNormByInternalId(*data2), _MM_HINT_T0);////////////	
                        //_mm_prefetch(getNormByInternalId(*data3), _MM_HINT_T0);////////////								
						}       
							   
                        if (!(visited_array[candidate_id] == visited_array_tag)) {
							
							//printf("ep_id = %d, wrong_id = %d\n", ep_id, candidate_id);
							visited_array[candidate_id] = visited_array_tag;

			                float* norm_pointer = (float*)getNormByInternalId(candidate_id);
			                //float v_norm = *norm_pointer;
			                //norm_pointer += 1;
			                float v_scale = *norm_pointer;
			                norm_pointer += 1;	
                            float ip =  dot_product_avx512_int16(query_point_int16, (int16_t*)norm_pointer, q_scale, v_scale);
                            float dist = 0.5 - ip;     

                            if (dist >= retset[retset_size-1].distance ) {
                 		        NeighborIndex nn3(candidate_id, dist, ip, true);

                                retset3[head3] = nn3;
							    if(size3 < LL3)
							        size3++;								
								head3 = (head3 + 1) % LL3; 
								continue;
							}

							if(retset[retset_size-1].flag == true){
                                retset2[head2] = retset[retset_size-1]; 
							    if(size2 < LL2)
							        size2++;									
								head2 = (head2 + 1) % LL2; 
							}
							else{
								top_candidates.emplace((retset[retset_size-1].distance + 0.5)*2, retset[retset_size-1].id);
							}
  
                 		    NeighborIndex nn2(candidate_id, dist, ip, true);
                            int r = InsertIntoPoolIndex(retset.data(), retset_size, nn2); 
                            
                            if(r <= next_next_k){
								if(r <= next_k){
									next_next_k = next_k + 1;
									next_k = r;
                                    _mm_prefetch((char *) (get_linklist0(candidate_id)), _MM_HINT_T0);														
								}
								else{
									next_next_k = r;
								}
							}				
						}
					}						

				    k = next_k;
				    next_k = next_next_k;
				}

			    for(int i = 0; i < retset_size; i++){
			        top_candidates.emplace((retset[i].distance + 0.5)*2, retset[i].id);
			    } 		
	
                std::sort(retset3.begin(), retset3.begin() + size3);
                std::rotate(retset2.begin(), retset2.begin() + head2, retset2.begin() + size2); 
                std::reverse(retset2.begin(), retset2.begin() + size2);

                {
                    int ii = 0, jj = 0, kk = 0;

                    while (kk < step && ii < size2 && jj < size3) {
                        if (retset2[ii].distance < retset3[jj].distance) {
                            retset[kk++] = retset2[ii++];
                        } else {
                            retset[kk++] = retset3[jj++];
                        }
                    }

                    while (kk < step && ii < size2) {
                        retset[kk++] = retset2[ii++];
                    }

                    while (kk < step && jj < size3) {
                        retset[kk++] = retset3[jj++];
                    }

                    retset_size = kk;	   
    
                    int w = 0;
                    int p2 = size2 - 1; 
                    int p3 = size3 - 1; 

                    while (w < LL2 && p2 >= ii && p3 >= jj) {
                        if (retset2[p2].distance > retset3[p3].distance) {
                            retset2[w++] = retset2[p2--]; 
                        } else {
                            retset2[w++] = retset3[p3--]; 
                        }
                    }

                    while (w < LL2 && p2 >= ii) {
                        retset2[w++] = retset2[p2--];
                    }

                    while (w < LL2 && p3 >= jj) {
                        retset2[w++] = retset3[p3--];
                    }

				    head2 = w % LL2;
				    size2 = w;
                    head3 = 0;
			        size3 = 0;			
			    }
			}		

			for(int i = 0; i < retset_size; i++){
			    top_candidates.emplace((retset[i].distance + 0.5)*2, retset[i].id);
			} 				
            visited_list_pool_->releaseVisitedList(vl);
			delete[] query_extended_point;
			std::free(table);
			return top_candidates;			
		} 

        void searchKnn(float* query_point, float* query_extended_point, size_t K, std::vector<Neighbor>& final_result, float* table, int step) const{
			
			//float qnorm = sqrt(dot_product_avx512(query_point, query_point));
			
			//float qnorm_vec[padded_dim_];
			//for(int i = 0; i < padded_dim_; i++)
			//	qnorm_vec[i] = query_point[i];
			
			//float qscale = 32767.0f / find_max_abs(qnorm_vec);
			float qscale = 32767.0f / find_max_abs(query_point);
			
			int16_t query_point_int16[padded_dim_];
			for(int i = 0; i < padded_dim_; i++)
				query_point_int16[i] = float_to_int16(query_point[i], qscale);
						
			VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
		
			int max_round2 = ef_ / step; 			

            int LL = step;
			int LL2 = step;
			int LL3 = step;
		
            std::vector<Neighbor> retset(LL);
			std::vector<Neighbor> retset2(LL2); 
			std::vector<Neighbor> retset3(LL3);			
					
			int head2 = 0;
			int head3 = 0;
			int size2 = 0;
			int size3 = 0;
			
			int cur_round = 0;
            char* index_pointer;
				
            int max_round = (maxM0_ / 16);		

            int next_id = center_neighbor[0];

            char* cur_pos0 = init_vec_;			
			for(int i = 0; i < step; i++){
			    int cur_id = next_id;
				
				if(i != step - 1){
				    next_id = center_neighbor[i+1];
                   _mm_prefetch((char *) (visited_array + next_id), _MM_HINT_T0);				
				}
				
				//float cur_norm = *(float*)cur_pos0;
				//cur_pos0 += 4;
				float cur_scale = *(float*)cur_pos0;
				cur_pos0 += 4;
				
				float cur_ip = dot_product_avx512_int16(query_point_int16, (int16_t*) cur_pos0, qscale, cur_scale);
				//float cur_dist = 0.5 - cur_ip;
				float cur_dist = -cur_ip;
				
				cur_pos0 += (padded_dim_ * sizeof(int16_t));
				
			    cur_round = max_round - 1;
			    for(int i = 1; i < max_round; i++){
				    if (cur_id < num_edge_offset[i]){
					    cur_round = i-1;
					    break;
				    }
			    }
                index_pointer = (data_level0_memory_query_ + size_links_offset_[cur_round] + 
				(cur_id - num_edge_offset[cur_round]) * (size_data_per_element_new_[cur_round]));

                retset[i] = Neighbor(*(unsigned int*)cur_pos0, cur_dist, cur_ip, true, index_pointer);
				cur_pos0 += 4;
				visited_array[cur_id] = visited_array_tag;			
			}
						
            std::sort(retset.begin(), retset.begin() + step);

            float* cur_pos = projVecQuery;
            for(int j = 0; j < level_; j++){
                float* y = query_extended_point + j * subdim_;
                float* current_level_ptr = table + j * (2 * m_);
                for(int i = 0; i < m_; i++){  //warning has been reversed
                    //_mm_prefetch(reinterpret_cast<const char*>(projVec[j][i].data()), _MM_HINT_T0);
                    current_level_ptr[i] = -1 * fstsubfunc_((void*) y, (void*) cur_pos, sub_func_param_);
                    cur_pos += subdim_;
					current_level_ptr[i + m_] = -1 * current_level_ptr[i];
                }
            }			
				
            int k = 0;
			int next_k = 0;	
            int next_next_k = 0;
			int real_data[maxM0_];
			
			int count = 0;
			
            const __m128i m128_4 = _mm_set1_epi8(0x0F);
            while (k < LL) {
                retset[k].flag = false;
					
			    if(k == next_k){
					next_k = LL;
					for(int ii = k+1; ii < LL; ii++){
						if(retset[ii].flag == true){
							next_k = ii;
							_mm_prefetch((char *) (retset[next_k].data_ptr), _MM_HINT_T0);
							break;
						}
					}
			    }
				
				int* data = (int*)retset[k].data_ptr;							
				int size = *data;
				int* datal = data + 1;
						          
                int rem = size & 15;      
                int round = (size + 15) >> 4;
                int div = rem | 16;						
						
				__m512 temp_center = _mm512_set1_ps(retset[k].ip);
                unsigned char* char_pointer = (unsigned char*) (datal + 16*round);
                        						
				for(int rr = 0; rr < round; rr++){
					int round2 = level_ / 8;
					__m512 sum = _mm512_setzero_ps();

for (int rr2 = 0; rr2 < round2; rr2++) {
    __m512i all_levels = _mm512_loadu_si512((const __m512i*)char_pointer);
    const float* t_ptr = table + rr2 * 128;

    __m128i r0 = _mm512_extracti32x4_epi32(all_levels, 0);
    __m128i r0_lo = _mm_and_si128(r0, m128_4);
    __m128i r0_hi = _mm_and_si128(_mm_srli_epi64(r0, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r0_hi, r0_lo)), _mm512_load_ps(t_ptr + 0)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r0_hi, r0_lo)), _mm512_load_ps(t_ptr + 16)));

    __m128i r1 = _mm512_extracti32x4_epi32(all_levels, 1);
    __m128i r1_lo = _mm_and_si128(r1, m128_4);
    __m128i r1_hi = _mm_and_si128(_mm_srli_epi64(r1, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r1_hi, r1_lo)), _mm512_load_ps(t_ptr + 32)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r1_hi, r1_lo)), _mm512_load_ps(t_ptr + 48)));

    __m128i r2 = _mm512_extracti32x4_epi32(all_levels, 2);
    __m128i r2_lo = _mm_and_si128(r2, m128_4);
    __m128i r2_hi = _mm_and_si128(_mm_srli_epi64(r2, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r2_hi, r2_lo)), _mm512_load_ps(t_ptr + 64)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r2_hi, r2_lo)), _mm512_load_ps(t_ptr + 80)));

    __m128i r3 = _mm512_extracti32x4_epi32(all_levels, 3);
    __m128i r3_lo = _mm_and_si128(r3, m128_4);
    __m128i r3_hi = _mm_and_si128(_mm_srli_epi64(r3, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r3_hi, r3_lo)), _mm512_load_ps(t_ptr + 96)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r3_hi, r3_lo)), _mm512_load_ps(t_ptr + 112)));

    char_pointer += 64;
}					

uint16_t* bf16_ptr = (uint16_t*)char_pointer;

__m512i bf16_01 = _mm512_loadu_si512((const __m512i*)bf16_ptr);
bf16_ptr += 32;

__m256i bf16_0 = _mm512_castsi512_si256(bf16_01);                 // low 16
__m256i bf16_1 = _mm512_extracti64x4_epi64(bf16_01, 1);           // high 16

__m512i v0_i32 = _mm512_cvtepu16_epi32(bf16_0);
v0_i32 = _mm512_slli_epi32(v0_i32, 16);
__m512 v_float = _mm512_castsi512_ps(v0_i32);

sum = _mm512_fmadd_ps(v_float, temp_center, sum);

__m512i v1_i32 = _mm512_cvtepu16_epi32(bf16_1);
v1_i32 = _mm512_slli_epi32(v1_i32, 16);
__m512 v_mul = _mm512_castsi512_ps(v1_i32);

//__m256i bf16_2 = _mm256_loadu_si256((const __m256i*)bf16_ptr);
//bf16_ptr += 16;

//__m512i v2_i32 = _mm512_cvtepu16_epi32(bf16_2);
//v2_i32 = _mm512_slli_epi32(v2_i32, 16);
//__m512 v_add = _mm512_castsi512_ps(v2_i32);
sum = _mm512_mul_ps(sum, v_mul);

                    __mmask16 tail_mask;
                    if (rr == round - 1 && div != 16) {
                        tail_mask = (1 << div) - 1; 
                    } else {
                        tail_mask = 0xFFFF; 
                    }

                    __mmask16 condition_mask = _mm512_cmp_ps_mask(sum, _mm512_set1_ps(retset[LL-1].distance), _CMP_LE_OS);

                    __m512i v_ids = _mm512_loadu_si512((const __m512i*)datal);
                    __mmask16 final_mask = condition_mask & tail_mask;
                    _mm512_mask_compressstoreu_epi32(real_data + count, final_mask, v_ids);

                    int stored_count = _mm_popcnt_u32(final_mask); 							
                    count += stored_count;

                        if(rr == round-2 && count > 0){
                            _mm_prefetch((char *) (visited_array + *real_data), _MM_HINT_T0);
                            _mm_prefetch(vec_level0_memory_+ (*real_data) * sec_part_, _MM_HINT_T0);												 
						}

                    char_pointer = (unsigned char*) bf16_ptr;							
                    datal += 16;	
				}

				if(count == 0){
					k = next_k;
					continue;						
				}

                next_next_k = next_k;
                 
                size = count;
                count = 0;				
				datal = real_data;

                for (size_t j = 0; j < size; j++) {
					int candidate_id = datal[j];
					int* data2 = datal+j+1;
					int* data3 = datal+j+2;

                    _mm_prefetch((char *) (visited_array + *data2), _MM_HINT_T0);
                    _mm_prefetch(vec_level0_memory_ + (*data2) * sec_part_, _MM_HINT_T0);////////////	
                    _mm_prefetch(vec_level0_memory_ + (*data3) * sec_part_, _MM_HINT_T0);////////////								
					       
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
						visited_array[candidate_id] = visited_array_tag;
								
			            float* norm_pointer = (float*) getNormByInternalIdQuery(candidate_id);

			            //float true_norm = *norm_pointer; 
			            //norm_pointer += 1;

                        float vscale = *norm_pointer;
						norm_pointer += 1;
						float ip_ = dot_product_avx512_int16(query_point_int16, (int16_t*) norm_pointer, qscale, vscale);
						//float dist = 0.5 - ip_;
                        float dist = -ip_;

                        if (dist >= retset[LL-1].distance ) {

				            cur_round = max_round - 1;
					        for(int i = 1; i < max_round; i++){
						        if (candidate_id < num_edge_offset[i]){
							        cur_round = i-1;
							        break;
						        }
					        }
                            index_pointer = (data_level0_memory_query_ + size_links_offset_[cur_round] + 
					            (candidate_id - num_edge_offset[cur_round]) * (size_data_per_element_new_[cur_round]));

                 		    Neighbor nn3( *(int*)(norm_pointer+(padded_dim_/2)), dist, ip_, true, index_pointer);

                            retset3[head3] = nn3; 
							if(size3 < LL3)
							    size3++;
							
							head3 = (head3 + 1) % LL3;
							continue;
						}

						if(retset[LL-1].flag == true){
                            retset2[head2] = retset[LL-1]; 
							if(size2 < LL2)
							    size2++;
							
							head2 = (head2 + 1) % LL2;
						}

				        cur_round = max_round - 1;
					    for(int i = 1; i < max_round; i++){
						    if (candidate_id < num_edge_offset[i]){
							    cur_round = i-1;
							    break;
						    }
					    }
                        
						index_pointer = (data_level0_memory_query_ + size_links_offset_[cur_round] + 
					        (candidate_id - num_edge_offset[cur_round]) * (size_data_per_element_new_[cur_round]));

                 		Neighbor nn2(*(int*)(norm_pointer + (padded_dim_/2)), dist, ip_, true, index_pointer);
                        int r = InsertIntoPool2(retset.data(), LL, nn2);

                        if(r <= next_next_k){
							if(r <= next_k){
								next_next_k = next_k + 1;
								next_k = r;
                                _mm_prefetch((char *) (index_pointer), _MM_HINT_T0);
							}
							else{
								next_next_k = r;
							}
						}				
					}												
				}				
				k = next_k;
				next_k = next_next_k;				
			}
			
			final_result = retset;

            int retset_size;
		    if(max_round2 > 1){

                std::sort(retset3.begin(), retset3.begin() + size3);
                std::rotate(retset2.begin(), retset2.begin() + head2, retset2.begin() + size2); 
                std::reverse(retset2.begin(), retset2.begin() + size2);

                {
                    int ii = 0, jj = 0, kk = 0;

                    while (kk < step && ii < size2 && jj < size3) {
                        if (retset2[ii].distance < retset3[jj].distance) {
                            retset[kk++] = retset2[ii++];
                        } else {
                            retset[kk++] = retset3[jj++];
                        }
                    }

                    while (kk < step && ii < size2) {
                        retset[kk++] = retset2[ii++];
                    }

                    while (kk < step && jj < size3) {
                        retset[kk++] = retset3[jj++];
                    }

                    retset_size = kk;	   
    
                    int w = 0;
                    int p2 = size2 - 1; 
                    int p3 = size3 - 1; 

                    while (w < LL2 && p2 >= ii && p3 >= jj) {
                        if (retset2[p2].distance > retset3[p3].distance) {
                            retset2[w++] = retset2[p2--]; 
                        } else {
                            retset2[w++] = retset3[p3--]; 
                        }
                    }

                    while (w < LL2 && p2 >= ii) {
                        retset2[w++] = retset2[p2--];
                    }

                    while (w < LL2 && p3 >= jj) {
                        retset2[w++] = retset3[p3--];
                    }

				    head2 = w % LL2;
				    size2 = w;
                    head3 = 0;
			        size3 = 0;			
			    }			
			}
			
			for(int seg = 1; seg < max_round2-1; seg++){			
				if(retset_size < 1) break;
					
                k = 0;
			    next_k = 0;	
                next_next_k = 0;			

                while (k < retset_size) {
                    retset[k].flag = false;
					
			        if(k == next_k){
					    next_k = retset_size;
					    for(int ii = k+1; ii < retset_size; ii++){
						    if(retset[ii].flag == true){
							    next_k = ii;
							    _mm_prefetch((char *) (retset[next_k].data_ptr), _MM_HINT_T0);
							     break;
						    }
					    }
					}

					int* data = (int*)retset[k].data_ptr;							
					int size = *data;
					int* datal = data + 1;
						         
                    int rem = size & 15;      
                    int round = (size + 15) >> 4;
                    int div = rem | 16;						
						
					__m512 temp_center = _mm512_set1_ps(retset[k].ip);
                    unsigned char* char_pointer = (unsigned char*) (datal + 16*round);

								
					for(int rr = 0; rr < round; rr++){
						int round2 = level_ / 8;
						__m512 sum = _mm512_setzero_ps();

for (int rr2 = 0; rr2 < round2; rr2++) {
    __m512i all_levels = _mm512_loadu_si512((const __m512i*)char_pointer);
    const float* t_ptr = table + rr2 * 128;

    __m128i r0 = _mm512_extracti32x4_epi32(all_levels, 0);
    __m128i r0_lo = _mm_and_si128(r0, m128_4);
    __m128i r0_hi = _mm_and_si128(_mm_srli_epi64(r0, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r0_hi, r0_lo)), _mm512_load_ps(t_ptr + 0)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r0_hi, r0_lo)), _mm512_load_ps(t_ptr + 16)));

    __m128i r1 = _mm512_extracti32x4_epi32(all_levels, 1);
    __m128i r1_lo = _mm_and_si128(r1, m128_4);
    __m128i r1_hi = _mm_and_si128(_mm_srli_epi64(r1, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r1_hi, r1_lo)), _mm512_load_ps(t_ptr + 32)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r1_hi, r1_lo)), _mm512_load_ps(t_ptr + 48)));

    __m128i r2 = _mm512_extracti32x4_epi32(all_levels, 2);
    __m128i r2_lo = _mm_and_si128(r2, m128_4);
    __m128i r2_hi = _mm_and_si128(_mm_srli_epi64(r2, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r2_hi, r2_lo)), _mm512_load_ps(t_ptr + 64)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r2_hi, r2_lo)), _mm512_load_ps(t_ptr + 80)));

    __m128i r3 = _mm512_extracti32x4_epi32(all_levels, 3);
    __m128i r3_lo = _mm_and_si128(r3, m128_4);
    __m128i r3_hi = _mm_and_si128(_mm_srli_epi64(r3, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r3_hi, r3_lo)), _mm512_load_ps(t_ptr + 96)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r3_hi, r3_lo)), _mm512_load_ps(t_ptr + 112)));

    char_pointer += 64;
}
												
uint16_t* bf16_ptr = (uint16_t*)char_pointer;

__m512i bf16_01 = _mm512_loadu_si512((const __m512i*)bf16_ptr);
bf16_ptr += 32;

__m256i bf16_0 = _mm512_castsi512_si256(bf16_01);                 // low 16
__m256i bf16_1 = _mm512_extracti64x4_epi64(bf16_01, 1);           // high 16

__m512i v0_i32 = _mm512_cvtepu16_epi32(bf16_0);
v0_i32 = _mm512_slli_epi32(v0_i32, 16);
__m512 v_float = _mm512_castsi512_ps(v0_i32);

sum = _mm512_fmadd_ps(v_float, temp_center, sum);

__m512i v1_i32 = _mm512_cvtepu16_epi32(bf16_1);
v1_i32 = _mm512_slli_epi32(v1_i32, 16);
__m512 v_mul = _mm512_castsi512_ps(v1_i32);

//__m256i bf16_2 = _mm256_loadu_si256((const __m256i*)bf16_ptr);
//bf16_ptr += 16;

//__m512i v2_i32 = _mm512_cvtepu16_epi32(bf16_2);
//v2_i32 = _mm512_slli_epi32(v2_i32, 16);
//__m512 v_add = _mm512_castsi512_ps(v2_i32);
sum = _mm512_mul_ps(sum, v_mul);

                        __mmask16 tail_mask;
                        if (rr == round - 1 && div != 16) {
                            tail_mask = (1 << div) - 1; 
                        } else {
                            tail_mask = 0xFFFF; 
                        }

                        __mmask16 condition_mask = _mm512_cmp_ps_mask(sum, _mm512_set1_ps(retset[retset_size-1].distance), _CMP_LE_OS);

                        __m512i v_ids = _mm512_loadu_si512((const __m512i*)datal);
                        __mmask16 final_mask = condition_mask & tail_mask;
                        _mm512_mask_compressstoreu_epi32(real_data + count, final_mask, v_ids);

                        int stored_count = _mm_popcnt_u32(final_mask); 

                        count += stored_count;

                        if(rr == round-2 && count > 0){
                            _mm_prefetch((char *) (visited_array + *real_data), _MM_HINT_T0);
                            _mm_prefetch(vec_level0_memory_+ (*real_data) * sec_part_, _MM_HINT_T0);												 
						}
						
                        char_pointer = (unsigned char*) bf16_ptr;							
                        datal += 16;	
					}

					if(count == 0){
					    k = next_k;
					    continue;						
					}

                    next_next_k = next_k;
                 
                    size = count;
                    count = 0;				
				    datal = real_data;
					
                    for (size_t j = 0; j < size; j++) {
					    int candidate_id = datal[j];
					    int* data2 = datal+j+1;
						int* data3 = datal+j+2;
#ifdef USE_SSE

                        _mm_prefetch((char *) (visited_array + *data2), _MM_HINT_T0);
                        _mm_prefetch(vec_level0_memory_ + (*data2) * sec_part_,
                        _MM_HINT_T0);////////////	
                        _mm_prefetch(vec_level0_memory_ + (*data3) * sec_part_,
                        _MM_HINT_T0);////////////								

#endif							       
                        if (!(visited_array[candidate_id] == visited_array_tag)) {
							visited_array[candidate_id] = visited_array_tag;
								
			                float* norm_pointer = (float*) getNormByInternalIdQuery(candidate_id);

			                //float true_norm = *norm_pointer; 
			                //norm_pointer += 1;

                            float vscale = *norm_pointer;
							norm_pointer += 1;

							float ip_ = dot_product_avx512_int16(query_point_int16, (int16_t*) norm_pointer, qscale, vscale);
							float dist = -ip_;

                            if (dist >= retset[retset_size-1].distance ) {

				                cur_round = max_round - 1;
					            for(int i = 1; i < max_round; i++){
						            if (candidate_id < num_edge_offset[i]){
							            cur_round = i-1;
							            break;
						            }
					            }
                                index_pointer = (data_level0_memory_query_ + size_links_offset_[cur_round] + 
					            (candidate_id - num_edge_offset[cur_round]) * (size_data_per_element_new_[cur_round]));

                 		        Neighbor nn3(*(int*)(norm_pointer+(padded_dim_/2)), dist, ip_, true, index_pointer);

                                retset3[head3] = nn3;
							    if(size3 < LL3)
							        size3++;								
								head3 = (head3 + 1) % LL3; 
								continue;
							}

							if(retset[retset_size-1].flag == true){
                                retset2[head2] = retset[retset_size-1]; 
							    if(size2 < LL2)
							        size2++;									
								head2 = (head2 + 1) % LL2; 
							}
  
				            cur_round = max_round - 1;
					        for(int i = 1; i < max_round; i++){
						        if (candidate_id < num_edge_offset[i]){
							        cur_round = i-1;
							        break;
						        }
					        }
                            
							index_pointer = (data_level0_memory_query_ + size_links_offset_[cur_round] + 
					            (candidate_id - num_edge_offset[cur_round]) * (size_data_per_element_new_[cur_round]));

                 		    Neighbor nn2(*(int*)(norm_pointer+(padded_dim_/2)), dist, ip_, true, index_pointer);
                            int r = InsertIntoPool2(retset.data(), retset_size, nn2); 
                            
                            if(r <= next_next_k){
								if(r <= next_k){
									next_next_k = next_k + 1;
									next_k = r;
                                    _mm_prefetch((char *) (index_pointer), _MM_HINT_T0);														
								}
								else{
									next_next_k = r;
								}
							}				
						}
					}						

				    k = next_k;
				    next_k = next_next_k;
				}
		
			    for(int j = 0; j < retset_size; j++){
				    if(retset[j].distance < final_result[K-1].distance){
				 	    InsertIntoPool2(final_result.data(), K, retset[j]);
				    }
				    else{
					    break;
				    }
			    }
	
                std::sort(retset3.begin(), retset3.begin() + size3);

                std::rotate(retset2.begin(), retset2.begin() + head2, retset2.begin() + size2); 
                std::reverse(retset2.begin(), retset2.begin() + size2);

                {
                    int ii = 0, jj = 0, kk = 0;

                    while (kk < step && ii < size2 && jj < size3) {
                        if (retset2[ii].distance < retset3[jj].distance) {
                            retset[kk++] = retset2[ii++];
                        } else {
                            retset[kk++] = retset3[jj++];
                        }
                    }

                    while (kk < step && ii < size2) {
                        retset[kk++] = retset2[ii++];
                    }

                    while (kk < step && jj < size3) {
                        retset[kk++] = retset3[jj++];
                    }

                    retset_size = kk;	   
    
                    int w = 0;
                    int p2 = size2 - 1; 
                    int p3 = size3 - 1; 

                    while (w < LL2 && p2 >= ii && p3 >= jj) {
                        if (retset2[p2].distance > retset3[p3].distance) {
                            retset2[w++] = retset2[p2--]; 
                        } else {
                            retset2[w++] = retset3[p3--]; 
                        }
                    }

                    while (w < LL2 && p2 >= ii) {
                        retset2[w++] = retset2[p2--];
                    }

                    while (w < LL2 && p3 >= jj) {
                        retset2[w++] = retset3[p3--];
                    }

				    head2 = w % LL2;
				    size2 = w;
                    head3 = 0;
			        size3 = 0;			
			    }			
			}
            //---------------last round --------------------------			
            
			if(max_round2 > 1){	
				k = 0;
			    next_k = 0;	
                next_next_k = 0;			

                while (k < retset_size) {
                    retset[k].flag = false;
					
			        if(k == next_k){
					    next_k = retset_size;
					    for(int ii = k+1; ii < retset_size; ii++){
						    if(retset[ii].flag == true){
							    next_k = ii;
							    _mm_prefetch((char *) (retset[next_k].data_ptr), _MM_HINT_T0);
							     break;
						    }
					    }
					}

					int* data = (int*)retset[k].data_ptr;							
					int size = *data;
					int* datal = data + 1;
						         
                    int rem = size & 15;      
                    int round = (size + 15) >> 4;
                    int div = rem | 16;						
						
					__m512 temp_center = _mm512_set1_ps(retset[k].ip);
                    unsigned char* char_pointer = (unsigned char*) (datal + 16*round);

								
					for(int rr = 0; rr < round; rr++){
						int round2 = level_ / 8;
						__m512 sum = _mm512_setzero_ps();

for (int rr2 = 0; rr2 < round2; rr2++) {
    __m512i all_levels = _mm512_loadu_si512((const __m512i*)char_pointer);
    const float* t_ptr = table + rr2 * 128;

    __m128i r0 = _mm512_extracti32x4_epi32(all_levels, 0);
    __m128i r0_lo = _mm_and_si128(r0, m128_4);
    __m128i r0_hi = _mm_and_si128(_mm_srli_epi64(r0, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r0_hi, r0_lo)), _mm512_load_ps(t_ptr + 0)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r0_hi, r0_lo)), _mm512_load_ps(t_ptr + 16)));

    __m128i r1 = _mm512_extracti32x4_epi32(all_levels, 1);
    __m128i r1_lo = _mm_and_si128(r1, m128_4);
    __m128i r1_hi = _mm_and_si128(_mm_srli_epi64(r1, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r1_hi, r1_lo)), _mm512_load_ps(t_ptr + 32)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r1_hi, r1_lo)), _mm512_load_ps(t_ptr + 48)));

    __m128i r2 = _mm512_extracti32x4_epi32(all_levels, 2);
    __m128i r2_lo = _mm_and_si128(r2, m128_4);
    __m128i r2_hi = _mm_and_si128(_mm_srli_epi64(r2, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r2_hi, r2_lo)), _mm512_load_ps(t_ptr + 64)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r2_hi, r2_lo)), _mm512_load_ps(t_ptr + 80)));

    __m128i r3 = _mm512_extracti32x4_epi32(all_levels, 3);
    __m128i r3_lo = _mm_and_si128(r3, m128_4);
    __m128i r3_hi = _mm_and_si128(_mm_srli_epi64(r3, 4), m128_4);
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(r3_hi, r3_lo)), _mm512_load_ps(t_ptr + 96)));
    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(r3_hi, r3_lo)), _mm512_load_ps(t_ptr + 112)));

    char_pointer += 64;
}
					
uint16_t* bf16_ptr = (uint16_t*)char_pointer;

__m512i bf16_01 = _mm512_loadu_si512((const __m512i*)bf16_ptr);
bf16_ptr += 32;

__m256i bf16_0 = _mm512_castsi512_si256(bf16_01);                 // low 16
__m256i bf16_1 = _mm512_extracti64x4_epi64(bf16_01, 1);           // high 16

__m512i v0_i32 = _mm512_cvtepu16_epi32(bf16_0);
v0_i32 = _mm512_slli_epi32(v0_i32, 16);
__m512 v_float = _mm512_castsi512_ps(v0_i32);

sum = _mm512_fmadd_ps(v_float, temp_center, sum);

__m512i v1_i32 = _mm512_cvtepu16_epi32(bf16_1);
v1_i32 = _mm512_slli_epi32(v1_i32, 16);
__m512 v_mul = _mm512_castsi512_ps(v1_i32);

//__m256i bf16_2 = _mm256_loadu_si256((const __m256i*)bf16_ptr);
//bf16_ptr += 16;

//__m512i v2_i32 = _mm512_cvtepu16_epi32(bf16_2);
//v2_i32 = _mm512_slli_epi32(v2_i32, 16);
//__m512 v_add = _mm512_castsi512_ps(v2_i32);
sum = _mm512_mul_ps(sum, v_mul);

                        __mmask16 tail_mask;
                        if (rr == round - 1 && div != 16) {
                            tail_mask = (1 << div) - 1; 
                        } else {
                            tail_mask = 0xFFFF; 
                        }

                        __mmask16 condition_mask = _mm512_cmp_ps_mask(sum, _mm512_set1_ps(retset[retset_size-1].distance), _CMP_LE_OS);

                        __m512i v_ids = _mm512_loadu_si512((const __m512i*)datal);
                        __mmask16 final_mask = condition_mask & tail_mask;
                        _mm512_mask_compressstoreu_epi32(real_data + count, final_mask, v_ids);

                        int stored_count = _mm_popcnt_u32(final_mask); 

                        count += stored_count;

                        if(rr == round-2 && count > 0){
                            _mm_prefetch((char *) (visited_array + *real_data), _MM_HINT_T0);
                            _mm_prefetch(vec_level0_memory_+ (*real_data) * sec_part_, _MM_HINT_T0);												 
						}
						
                        char_pointer = (unsigned char*) bf16_ptr;							
                        datal += 16;	
					}

					if(count == 0){
					    k = next_k;
					    continue;						
					}

                    next_next_k = next_k;
                 
                    size = count;
                    count = 0;				
				    datal = real_data;
					
                    for (size_t j = 0; j < size; j++) {
					    int candidate_id = datal[j];
					    int* data2 = datal+j+1;
						int* data3 = datal+j+2;
#ifdef USE_SSE

                        _mm_prefetch((char *) (visited_array + *data2), _MM_HINT_T0);
                        _mm_prefetch(vec_level0_memory_ + (*data2) * sec_part_,
                        _MM_HINT_T0);////////////	
                        _mm_prefetch(vec_level0_memory_ + (*data3) * sec_part_,
                        _MM_HINT_T0);////////////								

#endif							       
                        if (!(visited_array[candidate_id] == visited_array_tag)) {
							visited_array[candidate_id] = visited_array_tag;
								
			                float* norm_pointer = (float*) getNormByInternalIdQuery(candidate_id);

			                //float true_norm = *norm_pointer; 
			                //norm_pointer += 1;
                            
							float vscale = *norm_pointer;
							norm_pointer += 1;

							float ip_ = dot_product_avx512_int16(query_point_int16, (int16_t*) norm_pointer, qscale, vscale);
							float dist = -ip_;

                            if (dist >= retset[retset_size-1].distance ) {
								continue;
							}
  
				            cur_round = max_round - 1;
					        for(int i = 1; i < max_round; i++){
						        if (candidate_id < num_edge_offset[i]){
							        cur_round = i-1;
							        break;
						        }
					        }
                            
							index_pointer = (data_level0_memory_query_ + size_links_offset_[cur_round] + 
					            (candidate_id - num_edge_offset[cur_round]) * (size_data_per_element_new_[cur_round]));

                 		    Neighbor nn2(*(int*)(norm_pointer+(padded_dim_/2)), dist, ip_, true, index_pointer);
                            int r = InsertIntoPool2(retset.data(), retset_size, nn2); 
                            
                            if(r <= next_next_k){
								if(r <= next_k){
									next_next_k = next_k + 1;
									next_k = r;
                                    _mm_prefetch((char *) (index_pointer), _MM_HINT_T0);														
								}
								else{
									next_next_k = r;
								}
							}				
						}
					}						

				    k = next_k;
				    next_k = next_next_k;
				}

			    for(int j = 0; j < retset_size; j++){
				    if(retset[j].distance < final_result[K-1].distance){
				 	    InsertIntoPool2(final_result.data(), K, retset[j]);
				    }
				    else{
					    break;
				    }
			    }
			}			
            visited_list_pool_->releaseVisitedList(vl);	
		}

        inline void permute(float* dst, float* src){
			for(int i = 0; i < extended_dim_; i++){
				int new_pos = permutation_[i];
                if(new_pos < padded_dim_)
				    dst[i] = src[new_pos];
				else
					dst[i] = 0.0f;				
			}
		}		

        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top(); //minus
                dist_t dist_to_query = -curent_pair.first;  //positive
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    					
					dist_t curdist = calc_dist_int16(second_pair.second, curent_pair.second);
					
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair); //minus
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }
         
        inline linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_[internal_id]);
        };
        
        inline linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        tableint mutuallyConnectNewElement(tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
			
		    getNeighborsByHeuristic2(top_candidates, M_);
			
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            std::vector<dist_t> selectedDist;			
            selectedNeighbors.reserve(M_);
            selectedDist.reserve(M_);			
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                selectedDist.push_back(top_candidates.top().first);				
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);
            
                setListCount(ll_cur,selectedNeighbors.size());
                
				int *data = (int *) (ll_cur + 1);
				
				//-------reallocate size-------------------
				if(level == 0){
                    int rr = (selectedNeighbors.size() + 15) / 16;			
				    if(rr > 1){
					    size_t new_size = init_size_ + (rr-1) * segment_size_;
					    data_level0_memory_[cur_c] = (char*)realloc(data_level0_memory_[cur_c], new_size);
				    	ll_cur = get_linklist0(cur_c);
						data = (int *) (ll_cur + 1);
					}	
				}
				//--------------------------------------
				
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
					int* data2 = data + idx;
					
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    *data2 = selectedNeighbors[idx];
                }
                if (level == 0){				
				    addProjValAll(cur_c, selectedDist);	             
				}					
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                auto &bucket = addNeighbors[selectedNeighbors[idx]];
                if (!bucket.empty()) {
#ifdef USE_SSE
                    _mm_prefetch((char*)bucket.data(), _MM_HINT_T0);
#endif
                }

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                int *data = (int *) (ll_other + 1);

                if (sz_link_list_other < Mcurmax) {
					
				    //-------reallocate size-------------------
					int obj_id = selectedNeighbors[idx];
				    if(level == 0 && sz_link_list_other % 16 == 0 && is_full[obj_id] == false){
                        int rr = sz_link_list_other / 16;				
					    size_t new_size = init_size_ + rr * segment_size_;
					    data_level0_memory_[obj_id] = (char*)realloc(data_level0_memory_[obj_id], new_size);					
				        ll_other = get_linklist0(obj_id);
						data = (int *) (ll_other + 1);
					}
				    //--------------------------------------					
					
				    int* data2 = data + sz_link_list_other; 

					*data2 = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);

					if(level == 0 && sz_link_list_other + 1 == Mcurmax){
					    is_full[obj_id] = true;
					}
						
					if (level == 0){				
				        addProjVal(selectedNeighbors[idx], sz_link_list_other, selectedDist[idx]);
						delete_neighbor(addNeighbors, selectedNeighbors[idx], cur_c);
					}							
                } else {
					
					dist_t d_max = calc_dist_int16(cur_c, selectedNeighbors[idx]);
													
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
						int* data2 = data + j;											
					
					    dist_t tmp_val = calc_dist_int16(*data2, selectedNeighbors[idx]);
				
  						candidates.emplace(tmp_val, *data2);                      
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    if(level > 0){
                        int indx = 0;
                        while (candidates.size() > 0) {
							int* data2 = data + indx;
                            *data2 = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }
                        setListCount(ll_other, indx);							
					}
                    else{
                        int indx = 0;
						std::vector<tableint> arr;
                        while (candidates.size() > 0) {
							tableint tmp_id = candidates.top().second;
							arr.push_back(tmp_id);
                            candidates.pop();
                            indx++;
                        }
						int cur_pos = 0;
						
						bool flag = false;
                        for(int i = 0; i < indx; i++){
							if(cur_c == arr[i]){
								flag = true;
								break;
							}
						}
						
						for(int i = 0; i < sz_link_list_other; i++){
							int cur_id = *(data + i);
                            for(int j = 0; j < indx; j++){
								if(cur_id == arr[j]){
									if(cur_pos == i){
										cur_pos++;
										break;
									}
									else{
                                        *(data + cur_pos) = cur_id;
                                        char* index_link = (char*)(data + maxM0_);										
										for(int l = 0; l < level_; l++){
                                            unsigned char s = *getQuanVal(index_link, i, l);
                                            unsigned char* dst = getQuanVal(index_link, cur_pos, l);

                                            unsigned char src4;
                                            if (i % 2 == 0)
                                                src4 = get_high4(s);
                                            else
                                                src4 = get_low4(s);

                                            if (cur_pos % 2 == 0)
                                                write_high4(dst, src4);
                                            else
                                                write_low4(dst, src4);
										}
					                
										*getTmpVal(index_link,cur_pos) = *getTmpVal(index_link,i);						
					                    *getTmpVal2(index_link,cur_pos) = *getTmpVal2(index_link,i);
						                //*getNeighborNorm(index_link,cur_pos) = *getNeighborNorm(index_link,i);
										//*getFinalVal(data, cur_pos) =  *getFinalVal(data, i);   					
										cur_pos++;  
									    break;
									}
								}
							}
						}					
						if(flag == true){
					        data[cur_pos] = cur_c;
							addProjVal(selectedNeighbors[idx], cur_pos, selectedDist[idx]);
						    delete_neighbor(addNeighbors, selectedNeighbors[idx], cur_c);
						}
                        setListCount(ll_other, indx);
					}
                }
            }
			
            return next_closest_entry_point;
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }

        void setEfc(int efc) {
            ef_construction_ = efc;
        }

        void saveIndex() {
            std::ofstream output(infoPath_, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, m_);
			writeBinaryPOD(output, level_);
			writeBinaryPOD(output, subdim_);
			
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, size_links_per_element_);			
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);
            writeBinaryPOD(output, sec_part_);

            writeBinaryPOD(output, total_size);
            writeBinaryPOD(output, index_size_new);
            writeBinaryPOD(output, vecdim_);
            writeBinaryPOD(output, padded_dim_);			
            writeBinaryPOD(output, extended_dim_);
            writeBinaryPOD(output, maxk_);
			
            int max_round = maxM0_ / 16; 			
            for(int i = 0; i < max_round; i++){
                writeBinaryPOD(output, num_edge_offset[i]); 
                writeBinaryPOD(output, size_links_offset_[i]);
                writeBinaryPOD(output, size_data_per_element_new_[i]);    				
			}

            for (int i = 0; i < level_; ++i) {
                for (int j = 0; j < m_; ++j) {
                    output.write(reinterpret_cast<char*>(projVec[i][j].data()), subdim_ * sizeof(float));
                }
            }

			//output.write(data_level0_memory_, total_size);
            output.write((char*)center_neighbor.data(), maxk_ * sizeof(int));
            output.write(init_vec_, maxk_ * sec_part_);
		
		    free(init_vec_);

            output.close();
        }

        void loadIndex(const char* path_index, SpaceInterface<dist_t> *s1, SpaceInterface<dist_t> *s2, SpaceInterface<dist_t> *s3, size_t max_elements_i=0) {
//          std::ifstream input(location, std::ios::binary);
            std::string folderPath(path_index);
            std::string fullPath;
            if (!folderPath.empty() && (folderPath.back() == '/' || folderPath.back() == '\\')) {
                indexPath_ = folderPath + "index.bin";
                infoPath_ = folderPath + "info.bin";				
            } else {
                indexPath_ = folderPath + "/index.bin";
                infoPath_ = folderPath + "/info.bin";				
            }            
			
			std::ifstream input(infoPath_, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, m_);		
						
			readBinaryPOD(input, level_);
			readBinaryPOD(input, subdim_);
			
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, size_links_per_element_);				
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);
            readBinaryPOD(input, maxM_);
			
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);	
			readBinaryPOD(input, sec_part_);

            readBinaryPOD(input, total_size);
            readBinaryPOD(input, index_size_new);
            readBinaryPOD(input, vecdim_);
            readBinaryPOD(input, padded_dim_);			
            readBinaryPOD(input, extended_dim_);
            readBinaryPOD(input, maxk_);
			
            int max_round = maxM0_ / 16; 
            num_edge_offset = new tableint[max_round]();
			size_links_offset_ = new size_t[max_round]();
			size_data_per_element_new_ = new size_t[max_round]();
			
            for(int i = 0; i < max_round; i++){
                readBinaryPOD(input, num_edge_offset[i]); 
                readBinaryPOD(input, size_links_offset_[i]); 
                readBinaryPOD(input, size_data_per_element_new_[i]);  				
			}
			
			projVecQuery = new float[(size_t)(level_) * m_* subdim_];

            float* cur_pos = projVecQuery;
            for (size_t i = 0; i < level_; ++i) {
                for (size_t j = 0; j < m_; ++j) {	
                    input.read(reinterpret_cast<char*>(cur_pos), subdim_ * sizeof(float));
                    cur_pos += subdim_;
				}
            }

            data_size_ = s1->get_data_size();
            fstdistfunc_ = s1->get_dist_func();
			fstipfunc_ = s2->get_dist_func();
            dist_func_param_ = s1->get_dist_func_param();

			fstsubfunc_ = s3->get_dist_func();
			sub_func_param_ = s3->get_dist_func_param();

            std::ifstream input2(indexPath_, std::ios::binary);
			data_level0_memory_query_ = (char *) malloc(total_size);
		    if (data_level0_memory_query_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input2.read(data_level0_memory_query_, total_size);		    
			input2.close();

		    vec_level0_memory_ = data_level0_memory_query_ + index_size_new;
			
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);
            ef_ = 10;

            center_neighbor.resize(maxk_);
            input.read((char*)center_neighbor.data(), maxk_ * sizeof(int));

            init_vec_ = (char*) malloc(maxk_ * sec_part_);
            input.read(init_vec_, maxk_ * sec_part_);
            input.close();
            return;
        }

        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        float calc_vec_adjust(float* obj_vec, float* cen_vec, float* diff_vec){		
			float* tmp_cen = new float[extended_dim_];
			for(int i = 0; i < extended_dim_; i++)
				tmp_cen[i] = cen_vec[i]; 

			float scale = 0;

			float cos0 = (float) (dot_product_avx512_extended(obj_vec, tmp_cen));
			float cos = (float) (cos0);
			scale = (float) (cos);
				

	        for(int i = 0; i < extended_dim_; i++){
			    tmp_cen[i] = tmp_cen[i] * scale;
			}
			scale = cos0;	
				
	        for(int i = 0; i < extended_dim_; i++){
				diff_vec[i] = obj_vec[i] - tmp_cen[i];
			}		
			
			delete[] tmp_cen;
			return scale;		
		}

        inline void restore(float* v, int id) const{
			float* norm_pointer = (float *)getNormByInternalId(id);
            //float norm = *norm_pointer;
			//norm_pointer += 1;
            float scale = *norm_pointer;
			norm_pointer += 1;
            int16_t* data_int16 = (int16_t*) norm_pointer;			

			for(int i = 0; i < padded_dim_; i++)
				v[i] = 1.0f*data_int16[i]/scale;										
		}       

        inline float calc_dist_int16(int id1, int id2) const{
			float* norm_pointer1 = (float*) getNormByInternalId(id1);
		    //float norm1 = *norm_pointer1;
			//norm_pointer1 += 1;
			float scale1 = *norm_pointer1;
			norm_pointer1 += 1;					
					
		    float* norm_pointer2 = (float*)getNormByInternalId(id2);					
			//float norm2 = *norm_pointer2;
			//norm_pointer2 += 1;
			float scale2 = *norm_pointer2;					
			norm_pointer2 += 1;
					
			float dist = 2 - 
				2*dot_product_avx512_int16((int16_t*)norm_pointer1,(int16_t*)norm_pointer2,scale1,scale2);							
		    
			return dist;
		} 

        inline float calc_dist_float_int16(float* v, int id) const{			
			//float qnorm = fstipfunc_(v, v, dist_func_param_);
			float* norm_pointer = (float*) getNormByInternalId(id);
			float scale = *norm_pointer;
			norm_pointer += 1;					
					
            float dist = 2-2*dot_product_avx512_f32_i16(v, (int16_t*)norm_pointer, scale);
			return dist;
		}

        void addProjVal(int id, int pos, float dist){
            bool cur_sign, max_sign;
			int cur_ip;
		    float max_sum;
		    unsigned char max_ip;
			
            float* diff_data = new float[extended_dim_];
			float* LSH_data;
            float* cen_vec = new float[extended_dim_];
            unsigned char* LSH_info = new unsigned char[level_];			

            int* data = (int *) get_linklist0(id);
			
            size_t size = getListCount((linklistsizeint*)data);
			
            int* datal = data + 1;

            //float center_norm = *(float*) getNormByInternalId(id);			
			float* center_data = new float[padded_dim_];
			restore(center_data, id);	
				
			float* extended_center = new float[extended_dim_];
			permute(extended_center, center_data);

			int a = *(datal + pos);
			
            //float obj_norm = *(float*) getNormByInternalId(a);
			float* obj_data = new float[padded_dim_];
			restore(obj_data, a);			

            float* extended_obj = new float[extended_dim_];
			permute(extended_obj, obj_data);

            for(int l = 0; l < extended_dim_; l++){
				diff_data[l] = extended_obj[l] - extended_center[l];
			}
				
            bool is_zero = false;
            bool is_edge = false; 
			
			float tmp_adjust = calc_vec_adjust(extended_obj, extended_center, diff_data);
				
            LSH_data = diff_data;
			float error_val = 0;
				
			for(int k = 0; k < level_; k++){
		        for(int j = 0; j < m_; j++){
			        float sum = 0;
			        for(int l = 0; l < subdim_; l++){
			            sum += LSH_data[k * subdim_ + l] * projVec[k][j][l];
			        }

				    if(sum < 0) {sum = -1.0f * sum; cur_ip = j + 8;}
				    else{cur_ip = j;}
				
				    if(j == 0) {max_sum = sum; max_ip = cur_ip;}
				    else{
					    if(sum > max_sum) {max_sum = sum; max_ip = cur_ip;}
				    }
		        }
				LSH_info[k] = max_ip;
                error_val += max_sum;					
			}
			
			for(int k = 0; k < level_; k++){
			    for(int l = 0; l < subdim_; l++){
				    int s;
				    float s1;
					if(LSH_info[k] >= 8){
						s = LSH_info[k] - 8;
						s1 = -1.0f;
					}
					else{
						s = LSH_info[k];
						s1 = 1.0f;
					}
			        cen_vec[k * subdim_ + l] =  projVec[k][s][l] * s1;
			    }					
		    }
			
            float tmp_last = dot_product_avx512_extended(cen_vec, extended_center);			
			float tmp_norm2 = dot_product_avx512_extended(LSH_data, LSH_data);	

                float stored_val1 = tmp_norm2 / error_val;
				if(stored_val1 < 0 || !std::isfinite(stored_val1)){
					stored_val1 = 0;
				}

                float stored_val2 = tmp_last - (error_val * tmp_adjust / tmp_norm2);
				
                if(!std::isfinite(stored_val2) || stored_val2 > 32767 || stored_val2 < -32767){
					stored_val2 = 0;
				}

                char* index_pos = (char*)(datal+maxM0_);

			    for(int j = 0; j < level_; j++){
                    unsigned char* dst = getQuanVal(index_pos, pos, j);
                    unsigned char val = LSH_info[j];   

                    if (pos % 2 == 0) {
                        *dst = (*dst & 0x0F) | (val << 4);
                    } else {
                        *dst = (*dst & 0xF0) | val;
                    }				
			    }	
				
	            uint32_t u = *(uint32_t*)&stored_val2;
                uint32_t lsb = (u >> 16) & 1;
                u += 0x7FFF + lsb;
                uint16_t bf16 = (uint16_t)(u >> 16);			
				*getTmpVal(index_pos, pos) = bf16;

	            u = *(uint32_t*)&stored_val1;
                lsb = (u >> 16) & 1;
                u += 0x7FFF + lsb;
                bf16 = (uint16_t)(u >> 16);					
				*getTmpVal2(index_pos, pos) = bf16;
				
				//float tmp_norm = normInfo[*getExternalLabeLp(datal[pos])];
	            //u = *(uint32_t*)&tmp_norm;
                //lsb = (u >> 16) & 1;
                //u += 0x7FFF + lsb;
                //bf16 = (uint16_t)(u >> 16);					
                //*getNeighborNorm(index_pos, pos) = bf16;

            delete[] extended_center;
            delete[] extended_obj;			
            delete[] LSH_info;	
			delete[] diff_data;
			delete[] cen_vec;
			
			delete[] obj_data;
			delete[] center_data;
		}

        void addProjValAll(int id, std::vector<dist_t>& selectedDist){
            bool cur_sign, max_sign;
			int cur_ip;
		    float max_sum;
		    unsigned char max_ip;
			
            float* diff_data = new float[extended_dim_];
			float* LSH_data;
            float* cen_vec = new float[extended_dim_];
            unsigned char* LSH_info = new unsigned char[level_];			

            int* data;	
            data = (int *) get_linklist0(id);
			
            size_t size = getListCount((linklistsizeint*)data);
			
            int* datal = data + 1;

			//float center_norm = *(float*) getNormByInternalId(id);
			float* center_data = new float[padded_dim_];
			restore(center_data, id);            
			
			float* extended_center = new float[extended_dim_];
			permute(extended_center, center_data);

            float* extended_obj = new float[extended_dim_];
			float* obj_data = new float[padded_dim_];
            for (int i = 0; i < size; i++) {
				int a = *(datal+i);
				
			    //float obj_norm = *(float*) getNormByInternalId(a);
			    restore(obj_data, a); 

			    permute(extended_obj, obj_data);

                for(int l = 0; l < extended_dim_; l++){
					diff_data[l] = extended_obj[l] - extended_center[l];
				}

				float tmp_adjust = calc_vec_adjust(extended_obj, extended_center, diff_data);
				
                LSH_data = diff_data;
				float error_val = 0;
				
			    for(int k = 0; k < level_; k++){
			
		            for(int j = 0; j < m_; j++){
			            float sum = 0;
			            for(int l = 0; l < subdim_; l++){
			                sum += LSH_data[k * subdim_ + l] * projVec[k][j][l];
			            }

				        if(sum < 0) {sum = -1.0f * sum; cur_ip = j + 8;}
				        else{cur_ip = j;}
				
				        if(j == 0) {max_sum = sum; max_ip = cur_ip;}
				        else{
					        if(sum > max_sum) {max_sum = sum; max_ip = cur_ip;}
				        }
		            }
				    LSH_info[k] = max_ip;
                    error_val += max_sum;					
			    }
		
				for(int k = 0; k < level_; k++){
			        for(int l = 0; l < subdim_; l++){
						int s;
						float s1;
						if(LSH_info[k] >= 8){
							s = LSH_info[k] - 8;
							s1 = -1.0f;
						}
						else{
							s = LSH_info[k];
							s1 = 1.0f;
						}
						
			            cen_vec[k * subdim_ + l] =  projVec[k][s][l] * s1;
			        }					
				}
				
                float tmp_last = dot_product_avx512_extended(cen_vec, extended_center);	
			    float tmp_norm2 = dot_product_avx512_extended(LSH_data, LSH_data);

                float stored_val1 = tmp_norm2 / error_val;
				if(stored_val1 < 0 || !std::isfinite(stored_val1) ){
					stored_val1 = 0;
				}

                float stored_val2 = tmp_last - (error_val * tmp_adjust / tmp_norm2);

                if(!std::isfinite(stored_val2) || stored_val2 > 32767 || stored_val2 < -32767){
					stored_val2 = 0;
				}
				
                char* index_pos = (char*)(datal+maxM0_);

			    for(int j = 0; j < level_; j++){
                    unsigned char* dst = getQuanVal(index_pos, i, j);
                    unsigned char val = LSH_info[j];   
                    if (i % 2 == 0) {
                        *dst = (*dst & 0x0F) | (val << 4);
                    } else {
                        *dst = (*dst & 0xF0) | val;
                    }				
			    }

	            uint32_t u = *(uint32_t*)&stored_val2;
                uint32_t lsb = (u >> 16) & 1;
                u += 0x7FFF + lsb;
                uint16_t bf16 = (uint16_t)(u >> 16);			
				*getTmpVal(index_pos, i) = bf16;

	            u = *(uint32_t*)&stored_val1;
                lsb = (u >> 16) & 1;
                u += 0x7FFF + lsb;
                bf16 = (uint16_t)(u >> 16);					
				*getTmpVal2(index_pos, i) = bf16;
				
				//float tmp_norm = normInfo[*getExternalLabeLp(datal[i])];
	            //u = *(uint32_t*)&tmp_norm;
                //lsb = (u >> 16) & 1;
                //u += 0x7FFF + lsb;
                //bf16 = (uint16_t)(u >> 16);					
                //*getNeighborNorm(index_pos, i) = bf16;				
				//*getTmpVal(index_pos, i) = stored_val2;
				//*getTmpVal2(index_pos, i) = stored_val1;
                //*getNeighborNorm(index_pos, i) = normInfo[*getExternalLabeLp(datal[i])];		
			}
			
            delete[] extended_center;
            delete[] extended_obj;	
            delete[] LSH_info;	
			delete[] diff_data;
			delete[] cen_vec;
			
			delete[] obj_data;
			delete[] center_data;			
		}

        void addPoint(const void *data_point, labeltype label) {//warning

            tableint cur_c = 0;
            {
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
            }

            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            //int curlevel = getRandomLevel(mult_);

            //element_levels_[cur_c] = curlevel;

            //std::unique_lock <std::mutex> templock(global);
            //int maxlevelcopy = maxlevel_;
            //if (curlevel <= maxlevelcopy)
            //    templock.unlock();
            //tableint currObj = enterpoint_node_;
            //tableint enterpoint_copy = enterpoint_node_;

            memset(data_level0_memory_[cur_c], 0, init_size_);

            //float vec_normalized[padded_dim_];
            float* cur_vec = (float*) data_point;
			//for(int j = 0; j < padded_dim_; j++)
			//	vec_normalized[j] = cur_vec[j] / (*norm);
		
		    char* norm_pointer = getNormByInternalId(cur_c);
            //memcpy(norm_pointer, norm, sizeof(float));
			//norm_pointer += 4;
			float* scale_pointer = (float*)norm_pointer;
			*scale_pointer = 32767.0f / find_max_abs(cur_vec);			
			norm_pointer += 4;
			int16_t* dst_vec = (int16_t*) norm_pointer;

			for(int j = 0; j < padded_dim_; j++)
                dst_vec[j] = float_to_int16( cur_vec[j], *scale_pointer);

            memcpy(dst_vec + padded_dim_, &label, sizeof(labeltype));

            //if (curlevel) {
            //    linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            //    if (linkLists_[cur_c] == nullptr)
            //        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            //    memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            //}

            if (cur_c > 0) {
               //for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {

                    //if(level == 0){
						
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;						
				if(cur_c < step_)
                    top_candidates = searchBaseLayerFast(cur_c, 0, data_point);						
				else{
                    top_candidates = searchBaseLayerFast2(cur_c, data_point);	
				}
					//}
	
                    mutuallyConnectNewElement(cur_c, top_candidates, 0);
                //}

            } 
			//else {
            //    enterpoint_node_ = 0;
            //    maxlevel_ = curlevel;
            //}

            //if (curlevel > maxlevelcopy) {
            //    enterpoint_node_ = cur_c;
            //    maxlevel_ = curlevel;
            //}
        };

        void completeEdge(int i){
			
			int* data = (int*) get_linklist0(i);
			size_t size = getListCount((linklistsizeint*)data);
            start_pos[i] = (int)size;

			int* datal = data + 1;			
			//float* cur_dist = getFinalVal(datal, 0);	
            const auto& bucket = addNeighbors[i];
  
	        if (bucket.empty() || size == maxM0_) return;
            for (const auto& nb : bucket) {
                int id = static_cast<int>(nb.id);

                bool exists = false;
                for (int j = 0; j < size; ++j) {
                    if (datal[j] == id) {
                        exists = true;
                        break;
                    }
                }

                if (!exists) {
					
                    float dist_to_query = calc_dist_int16(i, id);
					//float dist_to_query = fstdistfunc_(cur_data, getDataByInternalId(id), dist_func_param_);
					
					bool skip_flag = false;
					
					_mm_prefetch((char*)(get_linklist0(datal[0])), _MM_HINT_T0);
					
					for(int l = 0; l < size; l++){						
						if(l < size - 1){
							_mm_prefetch((char*)(getNormByInternalId(datal[l+1])), _MM_HINT_T0);
						    _mm_prefetch((char*)(get_linklist0(datal[l+1])), _MM_HINT_T0);
						}
						
					    float two_dist = calc_dist_int16(id, datal[l]);
						//float two_dist = fstdistfunc_(getDataByInternalId(id), getDataByInternalId(datal[l]), dist_func_param_);
						
						float cur_dist = calc_dist_int16(i, datal[l]); 
						if(two_dist <= dist_to_query && cur_dist <= dist_to_query){
			                int* neighbor_data = (int*) get_linklist0(datal[l]);
			                size_t neighbor_size = getListCount((linklistsizeint*)neighbor_data);
			                int* neighbor_datal = neighbor_data + 1;							
							
							for(int ll = 0; ll < neighbor_size; ll++){
								if(neighbor_datal[ll] == id){
									skip_flag = true;
									break;
								}
							}
						}
					}
					
					if(skip_flag == false){
                        datal[size++] = id;
					    //addProjVal(i, (int) (size-1), dist_to_query);
					}
                }

                if (size == maxM0_) {
                    break;
                }

            }
			
            for (const auto& nb : bucket) {
                if (size % 16 == 0) break;
                int id = static_cast<int>(nb.id);
                bool exists = false;
                for (int j = 0; j < size; ++j) {
                    if (datal[j] == id) {
                        exists = true;
                        break;
                    }
                }

                if (!exists) {
                    datal[size++] = id;
                    //addProjVal(i, (int)(size - 1), 0.0f);
                }
	        }				
            setListCount((linklistsizeint *)data, (unsigned short int)size);			
		}

        void expandSpace(int i){
			int* data = (int*) get_linklist0(i);
			size_t size = getListCount((linklistsizeint*)data);
			
            if( ((start_pos[i] + 15) / 16) != ((size + 15) / 16) ){						
			    int rr = (size + 15) / 16;
			    size_t new_size = init_size_ + (rr-1) * segment_size_;
			    data_level0_memory_[i] = (char*)realloc(data_level0_memory_[i], new_size);	
			}
		}

        void addEdgeProj(int i){	
			int* data = (int*) get_linklist0(i);
			size_t size = getListCount((linklistsizeint*)data);
		    for(int j = start_pos[i]; j < size; j++){
				addProjVal(i, j, -1.0f);
			}			
		}

        void compression(int vecsize){
			int max_group = maxM0_ / 16;   
            num_edge_group = new tableint[max_group]();
			
			tableint* convert_id = new tableint[vecsize];
			tableint* convert_id2 = new tableint[vecsize];

			int* indicator= new int[vecsize];
			
            for(int i = 0; i < vecsize; i++){
				tableint* data = get_linklist0(i);
				tableint size = *data;
				data += 1;
				
				int round = size / 16;
				int residue = size % 16;
				if(residue == 0)
                   round--;     

				indicator[i] = round; //from 0
				convert_id[i] = num_edge_group[round];
				num_edge_group[round]++;
			}
			
            num_edge_offset = new tableint[max_group]();			
            for(int i = 1; i < max_group; i++) {
                num_edge_offset[i] = num_edge_offset[i-1] + num_edge_group[i-1];
            }

            inverse_id = new int[vecsize];
            for(int i = 0; i < vecsize; i++){
				convert_id2[i] = convert_id[i] + num_edge_offset[indicator[i]];
				inverse_id[convert_id2[i]] = i;
			}			

			size_data_per_element_new_ = new size_t[max_group]();
            for(int i = 0; i < max_group; i++){
                size_data_per_element_new_[i] = sizeof(linklistsizeint) + (i+1) * 16 * (2 * sizeof(int16_t) + sizeof(tableint) + level_/2);
			}	    
			index_size_new = 0;
            for(int i = 0; i < max_group; i++){
				index_size_new += size_data_per_element_new_[i] * num_edge_group[i];
			}
			
			total_size = index_size_new + (vecsize_ * sec_part_);			

#pragma omp parallel for schedule(dynamic)				
            for(size_t i = 0; i < vecsize; i++){
				tableint* data = get_linklist0(i);
				tableint size = *data;
				data += 1;
				
				for(int j = 0; j < size; j++){
					tableint obj_id = *(data + j);
					*(data + j) = convert_id2[obj_id];
				}
			}	

            size_links_offset_ = new size_t[max_group]();
            for(int i = 1; i < max_group; i++){
                size_links_offset_[i] = size_links_offset_[i-1] + (size_data_per_element_new_[i-1] * num_edge_group[i-1]);
			}

            std::ofstream output(indexPath_, std::ios::binary);
            for(int i = 0; i < vecsize; i++){
				int cur_id = inverse_id[i];  						
				int round = indicator[cur_id]; //0
                tableint pos = convert_id[cur_id];
                char* index_src = (char*) get_linklist0(cur_id);
				tableint size = *get_linklist0(cur_id);
				int residue = size % 16;
				int round2 = size / 16;
			    if(residue != 0) 
					round2 += 1;
				
				size_t tmp_segment = 4 + 4 * 16 * (round+1);
                output.write(index_src, tmp_segment);						
				index_src += 4 * (maxM0_+1);
				char* cen_data = index_src;
                output.write(cen_data, round2*(8 * level_ + 16*(2*sizeof(int16_t))));			
			}

            for(int i = 0; i < vecsize; i++){
				int cur_id = inverse_id[i];  			
				char* vec_src = (char*) getNormByInternalId(cur_id);				
                output.write(vec_src, sec_part_);							
			}
			
            output.close();
    
            for(int i = 0; i < maxk_; i++){
				center_neighbor[i] = convert_id2[center_neighbor[i]];
			}
    
			delete[] convert_id;
			delete[] convert_id2;
            delete[] indicator;	
            delete[] inverse_id;
			delete[] is_full;
    
            label_offset_ = data_size_ + sizeof(float);
			
        }

        void findCenterNeighbor(float *data_point) {

            int* others = new int[maxk_];
            init_vec_ = (char*) malloc(maxk_ * sec_part_);
			
			char* dst_adds = init_vec_;

            std::vector<unsigned int>result_ids(vecsize_);
            std::vector<Neighbor2> distances(vecsize_);
            #pragma omp parallel for
            for (size_t j = 0; j < vecsize_; j++) {			    
				float* cur_vec = new float[padded_dim_];
				restore(cur_vec, j);
                float tmp_dist = fstdistfunc_(cur_vec, data_point, dist_func_param_);
                distances[j] = Neighbor2{static_cast<unsigned int>(j), tmp_dist};
                delete[] cur_vec;
			}

            std::sort(distances.begin(), distances.end(), [](const Neighbor2& a, const Neighbor2& b){
                return a.distance < b.distance;
            });

            for (size_t j = 0; j < vecsize_; j++) {
                result_ids[j] = distances[j].id;
            }
			
            int count = 0;
			int count2 = 0;
			
			std::vector<std::vector<float>> kept_vectors;
            kept_vectors.reserve(maxk_);
			
            for (const auto& nb : result_ids) {
                size_t cid = nb;
                
				float* v = new float[padded_dim_];
				restore(v, (int)cid);
				
                std::vector<float> v_diff(padded_dim_);
                for (int d = 0; d < padded_dim_; d++)
                    v_diff[d] = v[d] - data_point[d];

                float norm_diff2 = fstipfunc_(v_diff.data(), v_diff.data(), dist_func_param_);
                float norm_diff = std::sqrt(norm_diff2);
                if (norm_diff < 1e-8f) continue;
                
				for (int d = 0; d < padded_dim_; d++)
                    v_diff[d] /= norm_diff;

                bool ok = true;
                for (const auto& u : kept_vectors) {
                    float dot = fstipfunc_(u.data(), v_diff.data(), dist_func_param_);
                    if (dot >= 0.5f) { // ≤60°
                        ok = false;
                        break;
                    }
                }

                if (ok) {
					center_neighbor.emplace_back(cid);
					char* src_adds = getNormByInternalId(cid);
					memcpy(dst_adds, src_adds, sec_part_);
					dst_adds += sec_part_;
                    count++;

					if(count >= maxk_)
						break;					
					
                    kept_vectors.emplace_back(std::move(v_diff));
                }
				else{
					if(count2 < maxk_){
					    others[count2] = cid;
					    count2++;
					}
				}			
				delete[] v;
            }
			
			if(count < maxk_){
				for(int j = 0; ; j++){
					char* src_adds = getNormByInternalId(others[j]);
					memcpy(dst_adds, src_adds, sec_part_);
					dst_adds += sec_part_;
					count++;
					if(count >= maxk_)
						break;							
				}					
			}			

            delete[] others;
            printf("#initial nodes = %d\n", count);			
        };		
    };
}
