// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef DFPBRT_UTIL_ERROR_H
#define DFPBRT_UTIL_ERROR_H

#include <dfpbrt/dfpbrt.h>


#include <string>
#include <string_view>

namespace dfpbrt {

// FileLoc Definition
struct FileLoc {
    FileLoc() = default;
    FileLoc(std::string_view filename) : filename(filename) {}
    std::string ToString() const;

    std::string_view filename;
    int line = 1, column = 0;
};


// Error Reporting Function Declarations
void Warning(const FileLoc *loc, const char *message);

template <typename... Args>
inline void Warning(const char *fmt, Args &&...args);

// Error Reporting Inline Functions
template <typename... Args>
inline void Warning(const FileLoc *loc, const char *fmt, Args &&...args) {
    Warning(loc, std::format(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
inline void Warning(const char *fmt, Args &&...args) {
    Warning(nullptr, std::format(fmt, std::forward<Args>(args)...).c_str());
}


}  // namespace pbrt

#endif  // PBRT_UTIL_ERROR_H
