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
  #define CHECK_EQ(a, b) CHECK_IMPL(a, b, ==)
  #define CHECK_NE(a, b) CHECK_IMPL(a, b, !=)
  #define CHECK_GT(a, b) CHECK_IMPL(a, b, >)
  #define CHECK_GE(a, b) CHECK_IMPL(a, b, >=)
  #define CHECK_LT(a, b) CHECK_IMPL(a, b, <)
  #define CHECK_LE(a, b) CHECK_IMPL(a, b, <=)

  // CHECK\_IMPL Macro Definition
#define CHECK_IMPL(a, b, op)                                                           \
    do {                                                                               \
        auto va = a;                                                                   \
        auto vb = b;                                                                   \
        if (!(va op vb))                                                               \
            LOG_FATAL("Check failed: %s " #op " %s with %s = %s, %s = %s", #a, #b, #a, \
                      va, #b, vb);                                                     \
    } while (false) /* swallow semicolon */

#endif 

  #define DCHECK(x) (CHECK(x))
  #define DCHECK_EQ(a, b) CHECK_EQ(a, b)
  #define DCHECK_NE(a, b) CHECK_NE(a, b)
  #define DCHECK_GT(a, b) CHECK_GT(a, b)
  #define DCHECK_GE(a, b) CHECK_GE(a, b)
  #define DCHECK_LT(a, b) CHECK_LT(a, b)
  #define DCHECK_LE(a, b) CHECK_LE(a, b)

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