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

namespace logging {

LogLevel logLevel = LogLevel::Error;
FILE *logFile;

}  // namespace logging

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

void Log(LogLevel level, const char *file, int line, const char *s) {
#ifdef DFPBRT_IS_GPU_CODE
    auto strlen = [](const char *ptr) {
        int len = 0;
        while (*ptr) {
            ++len;
            ++ptr;
        }
        return len;
    };

    // Grab a slot
    int offset = atomicAdd(&nRawLogItems, 1);
    GPULogItem &item = rawLogItems[offset % MAX_LOG_ITEMS];
    item.level = level;

    // If the file name is too long to fit in GPULogItem.file, then copy
    // the trailing bits.
    int len = strlen(file);
    if (len + 1 > sizeof(item.file)) {
        int start = len - sizeof(item.file) + 1;
        if (start < 0)
            start = 0;
        for (int i = start; i < len; ++i)
            item.file[i - start] = file[i];
        item.file[len - start] = '\0';

        // Now clobber the start with "..." to show it was truncated
        item.file[0] = item.file[1] = item.file[2] = '.';
    } else {
        for (int i = 0; i < len; ++i)
            item.file[i] = file[i];
        item.file[len] = '\0';
    }

    item.line = line;

    // Copy as much of the message as we can...
    int i;
    for (i = 0; i < sizeof(item.message) - 1 && *s; ++i, ++s)
        item.message[i] = *s;
    item.message[i] = '\0';
#else
    int len = strlen(s);
    if (len == 0)
        return;
    std::string levelString = (level == LogLevel::Verbose) ? "" : (ToString(level) + " ");

    // cut off everything up to pbrt/
    const char *fileStart = strstr(file, "pbrt/");
    std::string shortfile(fileStart ? (fileStart + 5) : file);

    if (logging::logFile) {
        fprintf(logging::logFile, "[ " LOG_BASE_FMT " %s:%d ] %s%s\n", LOG_BASE_ARGS,
                shortfile.c_str(), line, levelString.c_str(), s);
        fflush(logging::logFile);
    } else
        fprintf(stderr, "[ " LOG_BASE_FMT " %s:%d ] %s%s\n", LOG_BASE_ARGS,
                shortfile.c_str(), line, levelString.c_str(), s);
#endif
}

    void dfpbrt::Log_Fatal(LogLevel loglevel, const char *file, int line, std::string s){
        assert(loglevel == LogLevel::Fatal);
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        // cut off everything up to pbrt/
        const char *fileStart = strstr(file, "dfpbrt/");
        std::string shortfile(fileStart ? (fileStart + 5) : file);
        fprintf(stderr, "[ " LOG_BASE_FMT " %s:%d ] %s %s\n", LOG_BASE_ARGS,
                shortfile.c_str(), line, ToString(loglevel).c_str(), s.c_str());

        CheckCallbackScope::Fail();
        abort();
    }
}