#ifndef DFPBRT_UTIL_LOG_H
#define DFPBRT_UTIL_LOG_H

#include <dfpbrt/dfpbrt.h>

#include <string>
#include <vector>

namespace dfpbrt{
    //How to impl my own Log? here is an example
    //First, define log level, using enum
    enum class LogLevel { Verbose, Error, Fatal, Invalid }; 

    //Second, determine the interface of the Log function
    //What is logging? Write strings into a specific file and the formats of the strings are determined by the log-level
    void Log(LogLevel loglevel, const char * file, int line, const char * s);
    void Log_Fatal(LogLevel loglevel, const char * file, int line, const char * s);
    void Log_Error(LogLevel loglevel, const char * file, int line, const char * s);
    void Log_Verbose(LogLevel loglevel, const char * file, int line, const char * s);
    //Maros
    #define LOG_FATAL(...) dfpbrt::Log_Fatal(dfpbrt::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

    //utils
    std::string ToString(LogLevel level);


}

#endif