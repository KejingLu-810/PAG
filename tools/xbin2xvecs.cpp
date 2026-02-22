#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>
#include <cstring>

using namespace std;

// Buffer size: Process 100,000 lines of data per batch
const size_t BATCH_SIZE = 100000; 

// Judge string suffix
bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.size() < suffix.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Core conversion template function
// T: Data type (float or int32_t)
// type_name: Type name for log display
template <typename T>
int convert_fmt(const char* input_path, const char* output_path, const string& type_name) {
    ifstream in(input_path, ios::binary);
    if (!in) {
        cerr << "Error: Cannot open input file " << input_path << endl;
        return 1;
    }

    // 2. read Header (Global: N, D)
    // fbin: [N][D], ibin: [N][K]
    uint32_t n_rows = 0;
    uint32_t dim = 0;
    in.read((char*)&n_rows, 4);
    in.read((char*)&dim, 4);

    if (!in) {
        cerr << "Error: Failed to read header." << endl;
        return 1;
    }

    cout << "========================================" << endl;
    cout << "Mode: " << type_name << " conversion" << endl;
    cout << "  Input: " << input_path << endl;
    cout << "  Output: " << output_path << endl;
    cout << "  Count (N): " << n_rows << endl;
    cout << "  Dim/K (D): " << dim << endl;
    cout << "========================================" << endl;

    // 3. ofstream
    ofstream out(output_path, ios::binary);
    if (!out) {
        cerr << "Error: Cannot create output file " << output_path << endl;
        return 1;
    }

    // 4. split
    vector<T> buffer(BATCH_SIZE * dim);
    size_t processed = 0;
    
    // vec header -> int32 4B
    int32_t dim_header = (int32_t)dim;

    while (processed < n_rows) {
        size_t current_batch = min((size_t)BATCH_SIZE, (size_t)n_rows - processed);
        
        // read bins
        in.read((char*)buffer.data(), current_batch * dim * sizeof(T));
        if (!in && processed + current_batch < n_rows) {
             cerr << "Error: Unexpected end of input file." << endl;
             break;
        }

        // write vec format: [dim] [vector...]
        for (size_t i = 0; i < current_batch; ++i) {
            out.write((char*)&dim_header, sizeof(int32_t));
            out.write((char*)&buffer[i * dim], dim * sizeof(T));
        }

        processed += current_batch;
        
        // process
        if (processed % 100000 == 0 || processed == n_rows) {
            float progress = (float)processed / n_rows * 100.0f;
            cout << "\rProgress: " << processed << " / " << n_rows << " [" << (int)progress << "%]" << flush;
        }
    }

    cout << "\nDone.\n" << endl;
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
        cerr << "Supported formats:" << endl;
        cerr << "  .fbin -> .fvecs (float data)" << endl;
        cerr << "  .ibin -> .ivecs (int indices)" << endl;
        return 1;
    }

    string in_path = argv[1];
    
    if (ends_with(in_path, ".fbin")) {
        return convert_fmt<float>(argv[1], argv[2], "fbin->fvecs (Float)");
    } 
    else if (ends_with(in_path, ".ibin")) {
        return convert_fmt<int32_t>(argv[1], argv[2], "ibin->ivecs (Int)");
    } 
    else {
        cerr << "Unknown file extension. Please use .fbin or .ibin input files." << endl;
        return 1;
    }
}