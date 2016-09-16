#pragma once
#include "export.hpp"

#define DISCARD_MESSAGE true ? (void)0 : caffe::LogMessageVoidify() & caffe::eat_message().stream()

#ifndef __NVCC__
#include <boost/log/trivial.hpp>
// This file replaces google logging with boost logging and equivalent macros
#define LOG_EVERY_N_VARNAME(base, line) LOG_EVERY_N_VARNAME_CONCAT(base, line)
#define LOG_EVERY_N_VARNAME_CONCAT(base, line) base ## line

#define LOG_OCCURRENCES LOG_EVERY_N_VARNAME(occurrences_, __LINE__)
#define LOG_OCCURRENCES_MOD_N LOG_EVERY_N_VARNAME(occurrences_mod_n_, __LINE__)

#define LOG_EVERY_N(severity, n) \
    static int LOG_OCCURRENCES = 0, LOG_OCCURRENCES_MOD_N = 0; \
    ++LOG_OCCURRENCES; \
    if (++LOG_OCCURRENCES_MOD_N > n) LOG_OCCURRENCES_MOD_N -= n; \
    if (LOG_OCCURRENCES_MOD_N == 1) \
        LOG(severity)


#define LOG(severity) BOOST_LOG_TRIVIAL(severity) << "[" << __FUNCTION__ << "] "
#define LOG_IF(severity, condition) if(condition) LOG(severity)


#define LOG_FIRST_N(severity, n) static int LOG_OCCURRENCES = 0; if(LOG_OCCURRENCES <= n) ++LOG_OCCURRENCES; if(LOG_OCCURRENCES <= n) LOG(severity)

#define CHECK_OP(op, lhs, rhs) if(!(lhs op rhs)) caffe::throw_on_destroy(__FUNCTION__, __FILE__, __LINE__).stream()

#define CHECK_EQ(lhs, rhs)  CHECK_OP(==, lhs, rhs)
#define CHECK_NE(lhs, rhs)  CHECK_OP(!=, lhs, rhs)
#define CHECK_LE(lhs, rhs)  CHECK_OP(<=, lhs, rhs)
#define CHECK_LT(lhs, rhs)  CHECK_OP(< , lhs, rhs)
#define CHECK_GE(lhs, rhs)  CHECK_OP(>=, lhs, rhs)
#define CHECK_GT(lhs, rhs)  CHECK_OP(> , lhs, rhs)
#define CHECK(exp) if(!(exp)) caffe::throw_on_destroy(__FUNCTION__, __FILE__, __LINE__).stream()
#define CHECK_NOTNULL(val) \
  caffe::CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))
#ifdef _DEBUG

#define DLOG(sverity) LOG(sverity)
#define DCHECK_EQ(lhs, rhs)  CHECK_OP(==, lhs, rhs)
#define DCHECK_NE(lhs, rhs)  CHECK_OP(!=, lhs, rhs)
#define DCHECK_LE(lhs, rhs)  CHECK_OP(<=, lhs, rhs)
#define DCHECK_LT(lhs, rhs)  CHECK_OP(< , lhs, rhs)
#define DCHECK_GE(lhs, rhs)  CHECK_OP(>=, lhs, rhs)
#define DCHECK_GT(lhs, rhs)  CHECK_OP(> , lhs, rhs)
#define DCHECK(exp) if(!(exp)) caffe::throw_on_destroy(__FUNCTION__, __FILE__, __LINE__).stream()

#else

#define DLOG(severity) DISCARD_MESSAGE 
#define DCHECK_EQ(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK_NE(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK_LE(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK_LT(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK_GE(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK_GT(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK(exp) DISCARD_MESSAGE 

#endif

#else // __NVCC__

#define LOG_EVERY_N_VARNAME(base, line) DISCARD_MESSAGE
#define LOG_EVERY_N_VARNAME_CONCAT(base, line) DISCARD_MESSAGE

#define LOG_OCCURRENCES DISCARD_MESSAGE
#define LOG_OCCURRENCES_MOD_N DISCARD_MESSAGE

#define LOG(severity) DISCARD_MESSAGE

#define LOG_FIRST_N(severity, n) DISCARD_MESSAGE

#define CHECK_OP(op, lhs, rhs) DISCARD_MESSAGE

#define CHECK_EQ(lhs, rhs)  CHECK_OP(==, lhs, rhs)
#define CHECK_NE(lhs, rhs)  CHECK_OP(!=, lhs, rhs)
#define CHECK_LE(lhs, rhs)  CHECK_OP(<=, lhs, rhs)
#define CHECK_LT(lhs, rhs)  CHECK_OP(< , lhs, rhs)
#define CHECK_GE(lhs, rhs)  CHECK_OP(>=, lhs, rhs)
#define CHECK_GT(lhs, rhs)  CHECK_OP(> , lhs, rhs)
#define CHECK(exp) DISCARD_MESSAGE

#ifdef _DEBUG
#define DLOG(sverity) DISCARD_MESSAGE
#define DCHECK_EQ(lhs, rhs)  CHECK_OP(==, lhs, rhs)
#define DCHECK_NE(lhs, rhs)  CHECK_OP(!=, lhs, rhs)
#define DCHECK_LE(lhs, rhs)  CHECK_OP(<=, lhs, rhs)
#define DCHECK_LT(lhs, rhs)  CHECK_OP(< , lhs, rhs)
#define DCHECK_GE(lhs, rhs)  CHECK_OP(>=, lhs, rhs)
#define DCHECK_GT(lhs, rhs)  CHECK_OP(> , lhs, rhs)
#define DCHECK(exp) DISCARD_MESSAGE
#else

#define DLOG(severity) DISCARD_MESSAGE 
#define DCHECK_EQ(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK_NE(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK_LE(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK_LT(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK_GE(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK_GT(lhs, rhs)  DISCARD_MESSAGE 
#define DCHECK(exp) DISCARD_MESSAGE 
#endif
#endif // __NVCC__
#include <sstream>
#include <functional>
#include <string>
namespace caffe
{
    class DLL_EXPORT LogMessageVoidify {
     public:
      LogMessageVoidify() { }
      // This has to be an operator with a precedence lower than << but
      // higher than ?:
      void operator&(std::ostream&) { }
    };
    class DLL_EXPORT eat_message
    {
    public:
        eat_message(){}
        std::stringstream &stream(){return eat;}
    private:
        std::stringstream eat;
        eat_message(const eat_message&);
        void operator=(const eat_message&);
    };
    class DLL_EXPORT throw_on_destroy {
    public:
        throw_on_destroy(const char* function, const char* file, int line);
        std::ostringstream &stream();
        ~throw_on_destroy() throw();

    private:
        std::ostringstream log_stream_;
        throw_on_destroy(const throw_on_destroy&);
        void operator=(const throw_on_destroy&);
    };
    struct DLL_EXPORT IExceptionWithCallStackBase
	{
		virtual const char * CallStack() const = 0;
		virtual ~IExceptionWithCallStackBase() throw();
	};
    DLL_EXPORT void collect_callstack(size_t skipLevels, bool makeFunctionNamesStandOut, const std::function<void(const std::string&)>& write);
    DLL_EXPORT std::string print_callstack(size_t skipLevels, bool makeFunctionNamesStandOut);
    DLL_EXPORT std::string print_callstack(size_t skipLevels, bool makeFunctionNamesStandOut, std::stringstream& ss);
	// Exception wrapper to include native call stack string
	template <class E>
	class ExceptionWithCallStack : public E, public IExceptionWithCallStackBase
	{
	public:
		ExceptionWithCallStack(const std::string& msg, const std::string& callstack) :
			E(msg), m_callStack(callstack)
		{ }
		ExceptionWithCallStack(const E& exc, const std::string& callstack) :
			E(exc), m_callStack(callstack)
		{ }
        
		virtual const char * CallStack() const override { return m_callStack.c_str(); }

	protected:
		std::string m_callStack;
	};
    template<> class ExceptionWithCallStack<std::string>: public std::string, public IExceptionWithCallStackBase
    {
     	public:
		ExceptionWithCallStack(const std::string& msg, const std::string& callstack) :
			std::string(msg), m_callStack(callstack)
		{ }
        
		virtual const char * CallStack() const override { return m_callStack.c_str(); }

	protected:
		std::string m_callStack;
    };
    template <typename T>
    T* CheckNotNull(const char *file, int line, const char *names, T* t) {
    if (t == NULL) {
      std::stringstream ss;
      LOG(fatal) << "[" << file << ":" << line << "] " << names << "\nException at" << print_callstack(0, true, ss) << "\n";
      throw ExceptionWithCallStack<std::string>(std::string(names), ss.str());
    }
  return t;
}
}