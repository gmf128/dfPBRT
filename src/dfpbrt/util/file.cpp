#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/util/file.h>
#include <dfpbrt/util/check.h>
#include <dfpbrt/util/log.h>
#include <dfpbrt/util/string.h>

#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
// #ifndef PBRT_IS_WINDOWS
// #include <dirent.h>
// #include <fcntl.h>
// #include <sys/dir.h>
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <unistd.h>
// #endif

namespace dfpbrt{
std::vector<Float> ReadFloatFile(std::string filename) {
    FILE *f = FOpenRead(filename);
    if (f == nullptr) {
        //Error("%s: unable to open file", filename);
        return {};
    }

    int c;
    bool inNumber = false;
    char curNumber[32];
    int curNumberPos = 0;
    int lineNumber = 1;
    std::vector<Float> values;
    while ((c = getc(f)) != EOF) {
        if (c == '\n')
            ++lineNumber;
        if (inNumber) {
            if (curNumberPos >= (int)sizeof(curNumber))
                // LOG_FATAL("Overflowed buffer for parsing number in file: %s at "
                //           "line %d",
                //           filename, lineNumber);
            // Note: this is not very robust, and would accept something
            // like 0.0.0.0eeee-+--2 as a valid number.
            if ((isdigit(c) != 0) || c == '.' || c == 'e' || c == 'E' || c == '-' ||
                c == '+') {
                CHECK_LT(curNumberPos, sizeof(curNumber));
                curNumber[curNumberPos++] = c;
            } else {
                curNumber[curNumberPos++] = '\0';
                Float v;
                if (!Atof(curNumber, &v))
                    // ErrorExit("%s: unable to parse float value \"%s\"", filename,
                    //           curNumber);
                values.push_back(v);
                inNumber = false;
                curNumberPos = 0;
            }
        } else {
            if ((isdigit(c) != 0) || c == '.' || c == '-' || c == '+') {
                inNumber = true;
                curNumber[curNumberPos++] = c;
            } else if (c == '#') {
                while ((c = getc(f)) != '\n' && c != EOF)
                    ;
                ++lineNumber;
            } else if (isspace(c) == 0) {
                // Error("%s: unexpected character \"%c\" found at line %d.", filename, c,
                //       lineNumber);
                return {};
            }
        }
    }
    fclose(f);
    return values;
}

}