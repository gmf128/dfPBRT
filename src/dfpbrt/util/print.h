#ifndef DFPBRT_UTIL_PRINT_H
#define DFPBRT_UTIL_PRINT_H

#include <dfpbrt/dfpbrt.h>
#include <string>

namespace{
    // https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
inline std::string Red(const std::string &s) {
    const char *red = "\033[1m\033[31m";  // bold red
    const char *reset = "\033[0m";
    return std::string(red) + s + std::string(reset);
}

inline std::string Yellow(const std::string &s) {
    // https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    const char *yellow = "\033[1m\033[38;5;100m";
    const char *reset = "\033[0m";
    return std::string(yellow) + s + std::string(reset);
}

inline std::string Green(const std::string &s) {
    // https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    const char *green = "\033[1m\033[38;5;22m";
    const char *reset = "\033[0m";
    return std::string(green) + s + std::string(reset);
}
}

#endif