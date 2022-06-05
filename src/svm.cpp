#include <iostream>
#include "experiment.h"


int main(int argc, char** argv) {
    if (argc < 4) exit(1);
    std::string train(argv[1]), test(argv[2]), validate(argv[3]);
    const uint numa_nodes = config.get_numa_count();
    dataset train_dataset(numa_nodes, train);
    dataset test_dataset(numa_nodes, test);
    dataset validate_dataset(numa_nodes, validate);
    std::cout << "Loading completed!" << std::endl;

    std::string command;
    while (std::cin) {

        std::getline(std::cin, command);
        if (command.empty()) continue;
        if (command == "exit") break;

        experiment_configuration configuration(train_dataset, test_dataset, validate_dataset);
        if (!configuration.from_string(command)) {
            std::cerr << "Command failed to parse:\n" << command << std::endl;
            continue;
        }

        configuration.run_experiments();
    }

    return 0;
}
