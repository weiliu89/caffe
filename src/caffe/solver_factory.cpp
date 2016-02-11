#include "caffe/solver_factory.hpp"


using namespace caffe;


template<typename T> std::map<std::string, Solver<T>* (*)(const SolverParameter&)>& SolverRegistry<T>::Registry()
{
	static SolverRegistry<T>::CreatorRegistry* g_registry_ = new SolverRegistry<T>::CreatorRegistry();
	return *g_registry_;
}
template<typename Dtype> void SolverRegistry<Dtype>::AddCreator(const string& type, Creator creator) {
	SolverRegistry<Dtype>::CreatorRegistry& registry = Registry();
	CHECK_EQ(registry.count(type), 0)
		<< "Solver type " << type << " already registered.";
	registry[type] = creator;
}

template<typename Dtype> Solver<Dtype>* SolverRegistry<Dtype>::CreateSolver(const SolverParameter& param) {
	const string& type = param.type();
	SolverRegistry<Dtype>::CreatorRegistry& registry = Registry();
	CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
		<< " (known types: " << SolverTypeListString() << ")";
	return registry[type](param);
}
template<typename Dtype> vector<string> SolverRegistry<Dtype>::SolverTypeList() {
	SolverRegistry<Dtype>::CreatorRegistry& registry = Registry();
	vector<string> solver_types;
	for (typename SolverRegistry<Dtype>::CreatorRegistry::iterator iter = registry.begin();
		iter != registry.end(); ++iter) {
		solver_types.push_back(iter->first);
	}
	return solver_types;
}

template<typename Dtype> string SolverRegistry<Dtype>::SolverTypeListString() {
	vector<string> solver_types = SolverRegistry<Dtype>::SolverTypeList();
	string solver_types_str;
	for (vector<string>::iterator iter = solver_types.begin();
		iter != solver_types.end(); ++iter) {
		if (iter != solver_types.begin()) {
			solver_types_str += ", ";
		}
		solver_types_str += *iter;
	}
	return solver_types_str;
}


template class SolverRegistry < float >;
template class SolverRegistry < double>;
