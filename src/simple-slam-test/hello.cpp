#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "Hello World from C++!" << std::endl;
    
    for (int i = 1; i <= 5; i++) {
        std::cout << "Count: " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    std::cout << "Program finished!" << std::endl;
    return 0;
}