#include "log.h"
#include "check.h"
#include <dfpbrt/dfpbrt.h>

#include <iostream>
#include <cassert>
#include <mutex>

namespace dfpbrt{
    
    namespace {

float ElapsedSeconds() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    using clock = std::chrono::steady_clock;
    static clock::time_point start = clock::now();

    clock::time_point now = clock::now();
    int64_t elapseduS =
        std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
    return elapseduS / 1000000.;
}

uint32_t GetThreadIndex() {
    //Windows API
    return GetCurrentThreadId();
}

#define LOG_BASE_FMT "tid %03d @ %9.3fs"
#define LOG_BASE_ARGS GetThreadIndex(), ElapsedSeconds()

}  // namespace
    std::string ToString(LogLevel level) {
    switch (level) {
    case LogLevel::Verbose:
        return "VERBOSE";
    case LogLevel::Error:
        return "ERROR";
    case LogLevel::Fatal:
        return "FATAL";
    default:
        return "UNKNOWN";
    }
}

    void dfpbrt::Log_Fatal(LogLevel loglevel, const char *file, int line, const char* s){
        assert(loglevel == LogLevel::Fatal);
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        // cut off everything up to pbrt/
        const char *fileStart = strstr(file, "dfpbrt/");
        std::string shortfile(fileStart ? (fileStart + 5) : file);
        fprintf(stderr, "[ " LOG_BASE_FMT " %s:%d ] %s %s\n", LOG_BASE_ARGS,
                shortfile.c_str(), line, ToString(loglevel).c_str(), s);

        CheckCallbackScope::Fail();
        abort();
    }
}