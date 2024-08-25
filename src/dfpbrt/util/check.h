#ifndef DFPBRT_UTIL_CHECK_H
#define DFPBRT_UTIL_CHECK_H

#include <dfpbrt/util/log.h>
#include <functional>
#include <vector>

namespace dfpbrt{

    //CHECK(x) : x is a boolean expression whose value is either true or false. Here, if the value is false(cannot pass the check), then LOG_FATAL
    //(LOG_FATAL(...), true) is a comma expression, whose value is true(the latter one)
    //TODO: #define CHECK(x) (!(!(x) && (LOG_FATAL("Check failed: %s", #x), true)))
  #define CHECK(x) (!(!(x) && (LOG_FATAL(std::format("Check failed {:s}", #x)), true)))
    #define DCHECK(x) CHECK(x)

    // CheckCallbackScope Definition
class CheckCallbackScope {
  public:
    // CheckCallbackScope Public Methods
    CheckCallbackScope(std::function<std::string(void)> callback);

    ~CheckCallbackScope();

    CheckCallbackScope(const CheckCallbackScope &) = delete;
    CheckCallbackScope &operator=(const CheckCallbackScope &) = delete;

    static void Fail(){
        //todo: impl
    };

  private:
    // CheckCallbackScope Private Members
    static std::vector<std::function<std::string(void)>> callbacks;
};

}


#endif