#define NOMINMAX
#include "caffe/logging.hpp"
using namespace caffe;


#ifdef _WIN32
#include <windows.h>
#include "DbgHelp.h"
#include <WinBase.h>
#include <codecvt>
#pragma comment(lib, "Dbghelp.lib")
#endif
#include <algorithm>
#include <iostream>
#include <sstream>
// https://github.com/Microsoft/CNTK/blob/7c811de9e33d0184fdf340cd79f4f17faacf41cc/Source/Common/ExceptionWithCallStack.cpp

#ifdef _WIN32
typedef USHORT(WINAPI * CaptureStackBackTraceType)(__in ULONG, __in ULONG, __out PVOID*, __out_opt PULONG);
CaptureStackBackTraceType RtlCaptureStackBackTrace_func = nullptr;

// ----------------------------------------------------------------------------
// frequently missing Win32 functions
// ----------------------------------------------------------------------------

// strerror() for Win32 error codes
static inline std::wstring FormatWin32Error(DWORD error)
{
    wchar_t buf[1024] = {0};
    ::FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM, "", error, 0, buf, sizeof(buf) / sizeof(*buf) - 1, NULL);
    std::wstring res(buf);
    // eliminate newlines (and spaces) from the end
    size_t last = res.find_last_not_of(L" \t\r\n");
    if (last != std::string::npos)
        res.erase(last + 1, res.length());
    return res;
}

template <typename C>
struct basic_cstring : public std::basic_string<C>
{
    template <typename S>
    basic_cstring(S p)
        : std::basic_string<C>(p)
    {
    }
    operator const C*() const
    {
        return this->c_str();
    }
};

typedef basic_cstring<char> cstring;
typedef basic_cstring<wchar_t> wcstring;

#ifdef _MSC_VER

static inline cstring utf8(const std::wstring& p)
{
    return std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().to_bytes(p);
} // utf-16 to -8
static inline wcstring utf16(const std::string& p)
{
    return std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().from_bytes(p);
} // utf-8 to -16
#else // BUGBUG: we cannot compile the above on Cygwin GCC, so for now fake it using the mbs functions, which will only work for 7-bit ASCII strings
static inline std::string utf8(const std::wstring& p)
{
    return msra::strfun::wcstombs(p.c_str());
} // output: UTF-8... not really
static inline std::wstring utf16(const std::string& p)
{
    return msra::strfun::mbstowcs(p.c_str());
} // input: UTF-8... not really
#endif
static inline cstring utf8(const std::string& p)
{
    return p;
} // no conversion (useful in templated functions)
static inline wcstring utf16(const std::wstring& p)
{
    return p;
}

#else


#endif

static std::string MakeFunctionNameStandOut(std::string origName)
{
    try // guard against exception, since this is used for exception reporting
    {
        auto name = origName;
        // strip off modifiers for parsing (will be put back at the end)
        std::string modifiers;
        auto pos = name.find_last_not_of(" abcdefghijklmnopqrstuvwxyz");
        if (pos != std::string::npos)
        {
            modifiers = name.substr(pos + 1);
            name = name.substr(0, pos + 1);
        }
        bool hasArgList = !name.empty() && name.back() == ')';
        size_t angleDepth = 0;
        size_t parenDepth = 0;
        bool hitEnd = !hasArgList; // hit end of function name already?
        bool hitStart = false;
        // we parse the function name from the end; escape nested <> and ()
        // We look for the end and start of the function name itself (without namespace qualifiers),
        // and for commas separating function arguments.
        for (size_t i = name.size(); i--> 0;)
        {
            // account for nested <> and ()
            if (name[i] == '>')
                angleDepth++;
            else if (name[i] == '<')
                angleDepth--;
            else if (name[i] == ')')
                parenDepth++;
            else if (name[i] == '(')
                parenDepth--;
            // space before '>'
            if (name[i] == ' ' && i + 1 < name.size() && name[i + 1] == '>')
                name.erase(i, 1); // remove
            // commas
            if (name[i] == ',')
            {
                if (i + 1 < name.size() && name[i + 1] == ' ')
                    name.erase(i + 1, 1);  // remove spaces after comma
                if (!hitEnd && angleDepth == 0 && parenDepth == 1)
                    name.insert(i + 1, "  "); // except for top-level arguments, we separate them by 2 spaces for better readability
            }
            // function name
            if ((name[i] == '(' || name[i] == '<') &&
                parenDepth == 0 && angleDepth == 0 &&
                (i == 0 || name[i - 1] != '>') &&
                !hitEnd && !hitStart) // we hit the start of the argument list
            {
                hitEnd = true;
                name.insert(i, "  ");
            }
            else if ((name[i] == ' ' || name[i] == ':' || name[i] == '>') && hitEnd && !hitStart && i > 0) // we hit the start of the function name
            {
                if (name[i] != ' ')
                    name.insert(i + 1, " ");
                name.insert(i + 1, " "); // in total insert 2 spaces
                hitStart = true;
            }
        }
        return name + modifiers;
    }
    catch (...)
    {
        return origName;
    }
}



void caffe::collect_callstack(size_t skipLevels, bool makeFunctionNamesStandOut, const std::function<void(const std::string&)>& write)
{
    write("\n[CALL STACK]\n");

#ifdef _WIN32
    static const int MAX_CALLERS = 62;
    static const unsigned short MAX_CALL_STACK_DEPTH = 20;

    // RtlCaptureStackBackTrace() is a kernel API without default binding, we must manually determine its function pointer.
    if(RtlCaptureStackBackTrace_func == nullptr)
        RtlCaptureStackBackTrace_func  = (CaptureStackBackTraceType)(GetProcAddress(LoadLibrary("kernel32.dll"), "RtlCaptureStackBackTrace"));
    if (RtlCaptureStackBackTrace_func == nullptr) // failed somehow
        return write("Failed to generate CALL STACK. GetProcAddress(\"RtlCaptureStackBackTrace\") failed with error " + utf8(FormatWin32Error(GetLastError())) + "\n");

    HANDLE process = GetCurrentProcess();
    if (!SymInitialize(process, nullptr, TRUE))
        return write("Failed to generate CALL STACK. SymInitialize() failed with error " + utf8(FormatWin32Error(GetLastError())) + "\n");

    // get the call stack
    void* callStack[MAX_CALLERS];
    unsigned short frames;
    frames = RtlCaptureStackBackTrace_func(0, MAX_CALLERS, callStack, nullptr);

    SYMBOL_INFO* symbolInfo = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1); // this is a variable-length structure, can't use vector easily
    symbolInfo->MaxNameLen = 255;
    symbolInfo->SizeOfStruct = sizeof(SYMBOL_INFO);
    frames = std::min(frames, MAX_CALL_STACK_DEPTH);

    // format and emit
    size_t firstFrame = skipLevels + 1; // skip CollectCallStack()
    for (size_t i = firstFrame; i < frames; i++)
    {
        if (i == firstFrame)
            write("    > ");
        else
            write("    - ");

        if (SymFromAddr(process, (DWORD64)(callStack[i]), 0, symbolInfo))
        {
            write(makeFunctionNamesStandOut ? MakeFunctionNameStandOut(symbolInfo->Name) : symbolInfo->Name);
            write("\n");
        }
        else
        {
            DWORD error = GetLastError();
            char buf[17];
            sprintf_s(buf, "%p", callStack[i]);
            write(buf);
            write(" (SymFromAddr() error: " + utf8(FormatWin32Error(error)) + ")\n");
        }
    }

    write("\n");

    free(symbolInfo);

    SymCleanup(process);

#else // Linux

    unsigned int MAX_NUM_FRAMES = 1024;
    void* backtraceAddresses[MAX_NUM_FRAMES];
    unsigned int numFrames = backtrace(backtraceAddresses, MAX_NUM_FRAMES);
    char** symbolList = backtrace_symbols(backtraceAddresses, numFrames);

    for (size_t i = skipLevels; i < numFrames; i++)
    {
        char* beginName    = NULL;
        char* beginOffset  = NULL;
        char* beginAddress = NULL;

        // Find parentheses and +address offset surrounding the mangled name
        for (char* p = symbolList[i]; *p; ++p)
        {
            if (*p == '(')      // function name begins here
                beginName = p;
            else if (*p == '+') // relative address ofset
                beginOffset = p;
            else if ((*p == ')') && (beginOffset || beginName)) // absolute address
                beginAddress = p;
        }
        const int buf_size = 1024;
        char buffer[buf_size];

        if (beginName && beginAddress && (beginName < beginAddress))
        {
            *beginName++ = '\0';
            *beginAddress++ = '\0';
            if (beginOffset) // has relative address
                *beginOffset++ = '\0';

            // Mangled name is now in [beginName, beginOffset) and caller offset in [beginOffset, beginAddress).
            int status = 0;
            unsigned int MAX_FUNCNAME_SIZE = 4096;
            size_t funcNameSize = MAX_FUNCNAME_SIZE;
            char funcName[MAX_FUNCNAME_SIZE]; // working buffer
            const char* ret = abi::__cxa_demangle(beginName, funcName, &funcNameSize, &status);
            std::string fName;
            if (status == 0)
                fName = makeFunctionNamesStandOut ? MakeFunctionNameStandOut(ret) : ret; // make it a bit more readable
            else
                fName = beginName; // failed: fall back

            // name of source file--not printing since it is not super-useful
            //string sourceFile = symbolList[i];
            //static const size_t sourceFileWidth = 20;
            //if (sourceFile.size() > sourceFileWidth)
            //    sourceFile = "..." + sourceFile.substr(sourceFile.size() - (sourceFileWidth-3));
            while (*beginAddress == ' ') // eat unnecessary space
                beginAddress++;
            std::string pcOffset = beginOffset ? std::string(" + ") + beginOffset : std::string();
            snprintf(buffer, buf_size, "%-20s%-50s%s\n", beginAddress, fName.c_str(), pcOffset.c_str());
        }
        else // Couldn't parse the line. Print the whole line as it came.
            snprintf(buffer, buf_size, "%s\n", symbolList[i]);

        write(buffer);
    }

    free(symbolList);

#endif
}
std::string caffe::print_callstack(size_t skipLevels, bool makeFunctionNamesStandOut, std::stringstream& ss)
{
	caffe::collect_callstack(skipLevels, makeFunctionNamesStandOut, [&ss](const std::string& str) {ss << str; });
	return ss.str();
}
std::string caffe::print_callstack(size_t skipLevels, bool makeFunctionNamesStandOut)
{
    std::stringstream ss;
        
	return caffe::print_callstack(skipLevels, makeFunctionNamesStandOut, ss);
}

throw_on_destroy::throw_on_destroy(const char* function, const char* file, int line) 
{
    log_stream_ <<"[" << function << "-"
                << file << ":"
                << line << "]";
}
std::ostringstream &throw_on_destroy::stream()
{ 
    return log_stream_; 
}
throw_on_destroy::~throw_on_destroy() 
{
    std::stringstream ss;
    LOG(debug) << "\nException at" << print_callstack(0, true, ss) << "\n" << log_stream_.str();
	throw ExceptionWithCallStack<std::string>(log_stream_.str(), ss.str());
}
