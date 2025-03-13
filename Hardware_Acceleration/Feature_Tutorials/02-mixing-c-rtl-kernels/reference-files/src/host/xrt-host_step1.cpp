#include <iostream>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include "xrt/xrt_bo.h"
#include <experimental/xrt_xclbin.h>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define DATA_SIZE 4096

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <xclbin_file> <input_file> <output_file>\n";
        return -1;
    }

    std::string binaryFile = argv[1];
    std::string inputFile = argv[2];
    std::string outputFile = argv[3];

    // Open device and load xclbin
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(binaryFile);
    auto krnl = xrt::kernel(device, uuid, "krnl_vadd", xrt::kernel::cu_access_mode::exclusive);

    // Allocate buffers
    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    auto boIn1 = xrt::bo(device, vector_size_bytes, krnl.group_id(0));
    auto boIn2 = xrt::bo(device, vector_size_bytes, krnl.group_id(1));
    auto boOut = xrt::bo(device, vector_size_bytes, krnl.group_id(2));
    auto bo0_map = boIn1.map<int*>();
    auto bo1_map = boIn2.map<int*>();
    auto bo2_map = boOut.map<int*>();

    // Read input data
    std::ifstream inFile(inputFile);
    if (!inFile) {
        std::cerr << "Error: Unable to open input file.\n";
        return -1;
    }
    for (int i = 0; i < DATA_SIZE; ++i) {
        if (!(inFile >> bo0_map[i])) {
            std::cerr << "Error: Insufficient data in input file.\n";
            return -1;
        }
        bo1_map[i] = bo0_map[i]; // Modify if needed
    }
    inFile.close();

    // Sync input buffers to device
    boIn1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boIn2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Run the kernel
    auto run = krnl(boIn1, boIn2, boOut, DATA_SIZE);
    run.wait();

    // Sync output buffer from device
    boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Write output data
    std::ofstream outFile(outputFile);
    if (!outFile) {
        std::cerr << "Error: Unable to open output file.\n";
        return -1;
    }
    for (int i = 0; i < DATA_SIZE; ++i) {
        outFile << bo2_map[i] << "\n";
    }
    outFile.close();

    std::cout << "Execution completed successfully. Output written to " << outputFile << std::endl;
    return 0;
}
