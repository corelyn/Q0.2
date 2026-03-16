#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <map>

// ----------------------
//  Custom q0.2 quantizer
// ----------------------
uint8_t quantize_q0_2(float value) {
    if (value < -0.75f) value = -0.75f;
    if (value > 0.75f)  value = 0.75f;

    int q = static_cast<int>(std::round(value / 0.25f)); // -3..3
    uint8_t packed = static_cast<uint8_t>(q + 3);        // shift to 0..6
    if (packed > 3) packed = 3;                          // clamp to 2 bits
    return packed;
}

uint8_t pack_4x2bits(const uint8_t vals[4]) {
    return (vals[0] & 0x03) |
        ((vals[1] & 0x03) << 2) |
        ((vals[2] & 0x03) << 4) |
        ((vals[3] & 0x03) << 6);
}

// ----------------------
//  GGUF structures
// ----------------------
struct GGUFHeader {
    char magic[4];    // "GGUF"
    uint32_t version;
    uint32_t num_tensors;
};

struct TensorInfo {
    std::string name;
    uint64_t num_elements;
    uint64_t offset;
    uint8_t type;
};

// ----------------------
//  GGUF helper functions
// ----------------------
bool read_gguf_header(std::ifstream& fin, GGUFHeader& header) {
    fin.read(reinterpret_cast<char*>(&header), sizeof(GGUFHeader));
    if (!fin) return false;
    if (std::string(header.magic, 4) != "GGUF") return false;
    return true;
}

bool read_tensor_infos(std::ifstream& fin, uint32_t num_tensors, std::vector<TensorInfo>& tensors) {
    tensors.resize(num_tensors);
    for (uint32_t i = 0; i < num_tensors; ++i) {
        uint32_t name_len;
        fin.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        if (!fin) return false;

        tensors[i].name.resize(name_len);
        fin.read(reinterpret_cast<char*>(&tensors[i].name[0]), name_len);
        if (!fin) return false;

        fin.read(reinterpret_cast<char*>(&tensors[i].num_elements), sizeof(uint64_t));
        fin.read(reinterpret_cast<char*>(&tensors[i].offset), sizeof(uint64_t));
        fin.read(reinterpret_cast<char*>(&tensors[i].type), sizeof(uint8_t));
    }
    return true;
}

// ----------------------
//  Quantize tensor data
// ----------------------
std::vector<uint8_t> quantize_and_pack(const std::vector<float>& data) {
    std::vector<uint8_t> out;
    size_t n = data.size();
    for (size_t i = 0; i < n; i += 4) {
        uint8_t vals[4] = { 0,0,0,0 };
        for (size_t j = 0; j < 4 && (i + j) < n; ++j) {
            vals[j] = quantize_q0_2(data[i + j]);
        }
        out.push_back(pack_4x2bits(vals));
    }
    return out;
}

// ----------------------
//  Main
// ----------------------
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " input.gguf output.gguf\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    std::ifstream fin(input_file, std::ios::binary);
    if (!fin) { std::cerr << "Cannot open input GGUF\n"; return 1; }

    // Read header
    GGUFHeader header;
    if (!read_gguf_header(fin, header)) { std::cerr << "Invalid GGUF header\n"; return 1; }

    // Read tensor metadata
    std::vector<TensorInfo> tensors;
    if (!read_tensor_infos(fin, header.num_tensors, tensors)) {
        std::cerr << "Failed to read tensor metadata\n";
        return 1;
    }

    // Quantize all tensors
    std::map<std::string, std::vector<uint8_t>> packed_tensors;
    for (auto& tensor : tensors) {
        fin.seekg(static_cast<std::streamoff>(tensor.offset), std::ios::beg);

        std::vector<float> data(tensor.num_elements);
        fin.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(tensor.num_elements * sizeof(float)));
        if (!fin) { std::cerr << "Failed to read tensor data: " << tensor.name << "\n"; return 1; }

        packed_tensors[tensor.name] = quantize_and_pack(data);

        // Mark tensor type as custom q0.2
        tensor.type = 0xF0;
    }

    fin.close();

    // Write new GGUF
    std::ofstream fout(output_file, std::ios::binary);
    if (!fout) { std::cerr << "Cannot write output GGUF\n"; return 1; }

    // Write header
    fout.write(reinterpret_cast<char*>(&header), sizeof(GGUFHeader));

    // Write tensor metadata (with updated type)
    for (auto& tensor : tensors) {
        uint32_t name_len = static_cast<uint32_t>(tensor.name.size());
        fout.write(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        fout.write(reinterpret_cast<char*>(&tensor.name[0]), name_len);
        fout.write(reinterpret_cast<char*>(&tensor.num_elements), sizeof(uint64_t));
        fout.write(reinterpret_cast<char*>(&tensor.offset), sizeof(uint64_t));
        fout.write(reinterpret_cast<char*>(&tensor.type), sizeof(uint8_t));
    }

    // Write packed tensor data
    for (auto& tensor : tensors) {
        fout.write(reinterpret_cast<char*>(packed_tensors[tensor.name].data()),
            static_cast<std::streamsize>(packed_tensors[tensor.name].size()));
    }

    fout.close();

    std::cout << "GGUF q0.2 quantization complete!\n";
    return 0;
}
