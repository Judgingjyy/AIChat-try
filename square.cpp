#include <iostream>

int main() {
    double number;
    std::cout << "请输入一个数: ";
    std::cin >> number;
    double square = number * number;
    std::cout << number << " 的平方是: " << square << std::endl;
    return 0;
}
