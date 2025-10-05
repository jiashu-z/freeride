#include <iostream>
#include <filesystem>

int main() {
  std::string path = ".";
  try {
    if (std::filesystem::is_directory(path)) {
      for (const auto &entry : std::filesystem::directory_iterator(path)) {
        std::cout << entry.path() << std::endl;
        std::cout << entry.path().filename() << std::endl;
      }
    }
  } catch (const std::filesystem::filesystem_error &err) {
    std::cerr << err.what() << std::endl;
  }
  return 0;
}