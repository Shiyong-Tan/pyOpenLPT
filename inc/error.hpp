#pragma once
#include <exception>
#include <string>
#include <string_view>
#include <utility>

// ==============================
// Source location (C++20 or fallback)
// ==============================
#if __has_include(<source_location>) && __cplusplus >= 202002L
  #include <source_location>
  using SourceLoc = std::source_location;

  // Helper to get current source location
  inline constexpr SourceLoc current_loc(const SourceLoc& loc = SourceLoc::current()) { return loc; }

  // Macros for convenience (must be used at call sites)
  #define SRC_HERE ::current_loc()
  #define SRC_FILE(loc)   (loc.file_name())
  #define SRC_FUNC(loc)   (loc.function_name())
  #define SRC_LINE(loc)   (int(loc.line()))
#else
  // Fallback for pre-C++20 compilers
  struct SourceLoc {
    const char* file_name;
    const char* function_name;
    int         line;
  };

  // NOTE: __func__ is only valid inside a function. These macros MUST be expanded
  // at call sites (inside functions), not in global/member default initializers.
  #define SRC_HERE SourceLoc{__FILE__, __func__, __LINE__}
  #define SRC_FILE(loc)   ((loc).file_name)
  #define SRC_FUNC(loc)   ((loc).function_name)
  #define SRC_LINE(loc)   ((loc).line)
#endif

// ==============================
// Error codes (extend for your project)
// ==============================
enum class ErrorCode : int {
  Unknown            = 0,
  InvalidArgument    = 1,
  OutOfRange         = 2,
  UnsupportedType    = 3,
  IOfailure          = 4,
  NoEnoughData       = 5,
  GeometryFailure    = 6,
  ParallelGeometry   = 7,
  TotalInternalReflection = 8,
  NonConverged       = 9,
  NumericFailure     = 10,
  InvalidCameraState = 11,
};

// ==============================
/* Error container:
   - Do NOT default-initialize `where` with SRC_HERE here.
   - Always pass SRC_HERE from the call site (via macros below). */
struct Error {
  ErrorCode   code{ErrorCode::Unknown};  // Error category
  std::string message;                   // Human-readable message
  SourceLoc   where{};                   // Where the error occurred (set by caller)
  std::string context;                   // Optional extra context (key variables, etc.)

  Error() = default;

  // Constructor: caller must pass the source location explicitly.
  Error(ErrorCode c,
        std::string msg,
        SourceLoc loc,
        std::string ctx = {})
    : code(c),
      message(std::move(msg)),
      where(loc),
      context(std::move(ctx)) {}

  // Format as: "[code] message | ctx: ... @ file:line (func)"
  std::string toString() const {
    std::string out;
    out.reserve(message.size() + context.size() + 128);
    out += "[";
    out += std::to_string(int(code));
    out += "] ";
    out += message;
    if (!context.empty()) {
      out += " | ctx: ";
      out += context;
    }
    out += " @ ";
    out += SRC_FILE(where);
    out += ":";
    out += std::to_string(SRC_LINE(where));
    out += " (";
    out += SRC_FUNC(where);
    out += ")";
    return out;
  }
};

// ==============================
// Fatal exception (unrecoverable errors)
// ==============================
class FatalError : public std::exception {
public:
  explicit FatalError(Error err)
    : _err(std::move(err)), _what(_err.toString()) {}

  const char* what() const noexcept override { return _what.c_str(); }
  const Error& info() const noexcept { return _err; }

private:
  Error _err;
  std::string _what;
};

// ==============================
// Convenience macros (fatal)
// ==============================
// These macros ensure SRC_HERE is captured at the call site (inside a function).
#define THROW_FATAL(code, msg) \
  throw FatalError(Error{(code), std::string(msg), SRC_HERE, {}})

#define THROW_FATAL_CTX(code, msg, ctx) \
  throw FatalError(Error{(code), std::string(msg), SRC_HERE, std::string(ctx)})

// Assertions that throw on failure
#define REQUIRE(cond, code, msg) \
  do { if(!(cond)) THROW_FATAL((code), (msg)); } while(0)

#define REQUIRE_CTX(cond, code, msg, ctx) \
  do { if(!(cond)) THROW_FATAL_CTX((code), (msg), (ctx)); } while(0)


// ==============================
// Recoverable error channels
// ==============================

// Status: success/failure without a value
struct Status {
  Error err;       // Filled when ok == false
  bool  ok{true};  // True if success

  static Status OK() { return Status{Error{}, true}; }

  // Create a failure Status with location captured at the call site
  static Status From(ErrorCode c, std::string_view msg, std::string ctx = {}) {
    Status s;
    s.ok = false;
    s.err = Error{c, std::string(msg), SRC_HERE, std::move(ctx)};
    return s;
  }

  explicit operator bool() const { return ok; }
};

// StatusOr<T>: either holds T or an error
template <class T>
class StatusOr {
public:
  // Success constructor
  StatusOr(T value) : _ok(true), _value(std::move(value)) {}

  // Failure constructor (captures call-site location via Status::From)
  StatusOr(ErrorCode c, std::string_view msg, std::string ctx = {})
  : _ok(false), _status(Status::From(c, msg, std::move(ctx))) {}

  bool ok() const { return _ok; }
  explicit operator bool() const { return ok(); }

  const T& value() const { return _value; }
  T&       value()       { return _value; }

  const Status& status() const { return _status; }

private:
  bool    _ok{false};
  T       _value{};
  Status  _status{};
};

// ==============================
// Convenience macros for Status / StatusOr
// ==============================
#define STATUS_ERR(code, msg) \
  Status::From((code), (msg))

#define STATUS_ERR_CTX(code, msg, ctx) \
  Status::From((code), (msg), (ctx))

#define STATUS_OR_ERR(Ttype, code, msg) \
  StatusOr<Ttype>((code), (msg))

#define STATUS_OR_ERR_CTX(Ttype, code, msg, ctx) \
  StatusOr<Ttype>((code), (msg), (ctx))
