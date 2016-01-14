#ifndef INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_
#define INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_

#include "caffe/proto/caffe_pb.h"
#include "caffe/solver.hpp"
#include "caffe/common.hpp"
namespace caffe {

class DLL_EXPORT SignalHandler {
 public:
  // Contructor. Specify what action to take when a signal is received.
  SignalHandler(SolverAction::Enum SIGINT_action,
                SolverAction::Enum SIGHUP_action);
  ~SignalHandler();
  ActionCallback GetActionFunction();
 private:
  SolverAction::Enum CheckForSignals() const;
  SolverAction::Enum SIGINT_action_;
  SolverAction::Enum SIGHUP_action_;
};

}  // namespace caffe

#endif  // INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_
