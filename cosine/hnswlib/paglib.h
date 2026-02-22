#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>
//#include <boost/random.hpp>
//#include <boost/random/normal_distribution.hpp>

struct Neighbor {
    unsigned id;
    float distance;
    float ip;
    bool flag;
    const void* data_ptr; 

    Neighbor() = default;
    
    Neighbor(unsigned id, float distance, float ip, bool f, const void* ptr = nullptr) 
        : id{id}, distance{distance}, ip{ip}, flag(f), data_ptr(ptr) {}

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance;
    }
};

struct NeighborIndex {
    unsigned id;
    float distance;
    float ip;
    bool flag;

    NeighborIndex() = default;
    
    NeighborIndex(unsigned id, float distance, float ip, bool f) 
        : id{id}, distance{distance}, ip{ip}, flag(f) {}

    inline bool operator<(const NeighborIndex &other) const {
        return distance < other.distance;
    }
};

namespace hnswlib {
    //typedef size_t labeltype;
    typedef unsigned int labeltype;

    template <typename T>
    class pairGreater {
    public:
        bool operator()(const T& p1, const T& p2) {
            return p1.first > p2.first;
        }
    };

    template<typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
    }

    template<typename T>
    static void readBinaryPOD(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

    template<typename MTYPE>
    using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);


    template<typename MTYPE>
    class SpaceInterface {
    public:
        //virtual void search(void *);
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        virtual void *get_dist_func_param() = 0;

        virtual ~SpaceInterface() {}
    };

    template<typename dist_t>
    class AlgorithmInterface {
    public:
        virtual void addPoint(const void *datapoint, labeltype label)=0;
		virtual void setEfc(int) = 0;
        //virtual void getMaximalVal(int, float* , float*, float*)=0;
        virtual void findCenterNeighbor(float*)=0;			
		virtual void compression(int)=0;	
		virtual void completeEdge(int)=0;
		virtual void expandSpace(int)=0;	
		virtual void addEdgeProj(int)=0;			
        virtual void searchKnn(float *query_data, float *query_data2, size_t k, std::vector<Neighbor>& result, float* query_lsh, int step)const = 0;

         //Return k nearest neighbor in the order of closer fist
        virtual std::vector<std::pair<dist_t, labeltype>>
            searchKnnCloserFirst(const void* query_data, size_t k) const;

        virtual void saveIndex()=0;
        virtual ~AlgorithmInterface(){
        }
    };

    template<typename dist_t>
    std::vector<std::pair<dist_t, labeltype>>
    AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k) const {
        std::vector<std::pair<dist_t, labeltype>> result;

        // here searchKnn returns the result in the order of further first
		
		/*
        auto ret = searchKnn(query_data, k);
        {
            size_t sz = ret.size();
            result.resize(sz);
            while (!ret.empty()) {
                result[--sz] = ret.top();
                ret.pop();
            }
        }
        */

        return result;
    }

}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
