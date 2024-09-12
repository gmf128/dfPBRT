// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <dfpbrt/util/error.h>

#include <dfpbrt/util/check.h>
#include <dfpbrt/util/print.h>
#include <dfpbrt/util/log.h>


#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#ifdef DFPBRT_IS_WINDOWS
#include <windows.h>
#endif

namespace dfpbrt {

static bool quiet = false;


std::string FileLoc::ToString() const {
    return std::format("{}:{}:{}", std::string(filename.data(), filename.size()), line,
                        column);
}

static void processError(const char *errorType, const FileLoc *loc, const char *message) {
    // Build up an entire formatted error string and print it all at once;
    // this way, if multiple threads are printing messages at once, they
    // don't get jumbled up...
    std::string errorString = Red(errorType);

    if (loc)
        errorString += ": " + loc->ToString();

    errorString += ": ";
    errorString += message;

    // Print the error message (but not more than one time).
    static std::string lastError;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (errorString != lastError) {
        fprintf(stderr, "%s\n", errorString.c_str());
        LOG_VERBOSE(std::format("{}", errorString));
        lastError = errorString;
    }
}

void Warning(const FileLoc *loc, const char *message) {
    if (quiet)
        return;
    processError("Warning", loc, message);
}

void ErrorExit(const FileLoc *loc, const char *message) {
    processError("Error", loc, message);
    std::exit(EXIT_FAILURE);
    return;
}

void Error(const FileLoc *loc, const char *message){
    processError("Error", loc, message);
    return;
}


}  // namespace pbrt
