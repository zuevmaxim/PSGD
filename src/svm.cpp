#include <iostream>
#include <fstream>
#include "run_configuration.h"


int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Expected arguments are:\n"
                  << "1) train dataset path\n"
                  << "2) test dataset path\n"
                  << "3) validate dataset path\n"
                  << "4) output CSV file path\n"
                  << std::endl;
        exit(1);
    }
    std::string train(argv[1]), test(argv[2]), validate(argv[3]), output(argv[4]);
    const uint numa_nodes = config.get_numa_count();
    dataset train_dataset(numa_nodes, train);
    dataset test_dataset(numa_nodes, test);
    dataset validate_dataset(numa_nodes, validate);

    std::istream* in_ptr;
    std::ifstream input_file;
    if (argc > 5) {
        std::string input(argv[5]);
        input_file.open(input);
        if (!input_file.good()) {
            std::cerr << "Error opening input file " << input << std::endl;
            exit(2);
        }
        in_ptr = &input_file;
    } else {
        in_ptr = &std::cin;
    }
    std::istream& in = *in_ptr;

    std::ofstream output_file;
    output_file.open(output);
    if (!output_file.good()) {
        std::cerr << "Error opening output file " << output << std::endl;
        exit(3);
    }


    if (argc > 6 && strcmp("-v", argv[6]) == 0) {
        experiment_configuration::verbose = true;
    }

    std::cout << "Loading completed!" << std::endl;

    std::string command;
    while (in) {

        std::getline(in, command);
        if (command.empty()) continue;
        if (command == "exit") break;

        experiment_configuration configuration(train_dataset, test_dataset, validate_dataset, output_file);
        if (!configuration.from_string(command)) {
            std::cerr << "Command failed to parse:\n" << command << std::endl;
            continue;
        }

        configuration.run_experiments();
    }
    output_file.close();
    if (input_file.is_open()) input_file.close();

    return 0;
}
