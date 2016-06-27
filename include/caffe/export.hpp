#pragma once

#ifdef _MSC_VER
#define TEMPLATE_EXTERN
#if defined libcaffe_EXPORTS
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif
#else
#define DLL_EXPORT
#define TEMPLATE_EXTERN extern
#endif