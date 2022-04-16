#pragma once
#include "Intersector3d.h"
#include "collision6d.h"
#include "../TreeAdapter/TreeAdapter.h"
#include "../TreeAdapter/TreeAdapter6d.h"


template <typename T>
void dprint(const T& t) {
	for (const auto& a : t)
		std::cout << a << " - ";
	std::cout << '\n';
}

template <typename Array6dT>
class Intersector6d
{
public:
	typedef typename Array6dT::SpaceAdapter SpaceAdapter; // Rewrite with using?
	typedef typename Array6dT::VelocityAdapter VelocityAdapter;
	typedef decltype(__get_const_iterator<SpaceAdapter>()) SpaceIteratorT;
	typedef decltype(__get_const_iterator<VelocityAdapter>()) VelocityIteratorT;
private:
	Intersector3d<SpaceAdapter> inter_r_;
	Intersector3d<VelocityAdapter> inter_v_;
public:
	Intersector6d(size_t tree_depth) : inter_r_(tree_depth), inter_v_(tree_depth)
	{ }

	std::vector<collision6d<VelocityIteratorT>> intersect(
		const Array6dT& a, const Array6dT& b,
		double rTolerance, double vTolerance) {
		std::vector<collision6d<VelocityIteratorT>> res;
		// Intersect by space
		const auto& colls_r = inter_r_.intersect(SpaceAdapter(a), SpaceAdapter(b), rTolerance);
		for (const auto& coll_r : colls_r) {
			//dprint(coll_r.apoints);
			auto colls_v = inter_v_.intersect(VelocityAdapter(coll_r.apoints), VelocityAdapter(coll_r.bpoints), vTolerance);
			for (auto& coll_v : colls_v) {
				std::vector<VelocityIteratorT> apoints = std::move(coll_v.apoints);
				std::vector<VelocityIteratorT> bpoints = std::move(coll_v.bpoints);
				res.emplace_back(std::move(apoints), std::move(bpoints), coll_r.cube, coll_v.cube);
			}				
		}

		return res;
	}

	// Work only after calling .intersect()
	std::vector<double> get_real_precision(double target_tolerance) const
	{
		auto rtol = inter_r_.get_real_precision(target_tolerance);
		const auto vtol = inter_v_.get_real_precision(target_tolerance);
		
		rtol.insert(rtol.end(), vtol.begin(), vtol.end());

		return rtol;
	}
};


template <typename Array6dT>
class Intersector6dRonly
{
public:
	typedef typename Array6dT::SpaceAdapter SpaceAdapter; // Rewrite with using?
	typedef typename Array6dT::VelocityAdapter VelocityAdapter;
	typedef decltype(__get_const_iterator<SpaceAdapter>()) SpaceIteratorT;
private:
	Intersector3d<SpaceAdapter> intersector_;
public:
	Intersector6dRonly(size_t tree_depth) : intersector_(tree_depth)
	{ }

	std::vector<collision6d<SpaceIteratorT>> intersect(
		const Array6dT& a, const Array6dT& b, double atol) {
		std::vector<collision6d<SpaceIteratorT>> res;
		const std::vector<collision3d<SpaceIteratorT>> colls = intersector_.intersect(SpaceAdapter(a), SpaceAdapter(b), atol);

		for (size_t i = 0; i < colls.size(); ++i) {
			//auto vcube = ::get_boundary_cube3d(VelocityAdapter(colls[i].apoints), VelocityAdapter(colls[i].bpoints));
			cube3d vcube;
			std::vector<SpaceIteratorT> apoints = std::move(colls[i].apoints);
			std::vector<SpaceIteratorT> bpoints = std::move(colls[i].bpoints);
			res.emplace_back(std::move(apoints), std::move(bpoints), colls[i].cube, vcube);
		}

		return res;
	}

	// Work only after calling .intersect()
	std::vector<double> get_real_precision(double target_tolerance) const
	{
		auto rtol = intersector_.get_real_precision(target_tolerance);
		static const std::vector<double> vtol = {0.0, 0.0, 0.0}; 
		
		rtol.insert(rtol.end(), vtol.begin(), vtol.end());

		return rtol;
	}
};
