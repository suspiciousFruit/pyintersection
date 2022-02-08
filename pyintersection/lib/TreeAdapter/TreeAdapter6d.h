#pragma once
#include <vector>
#include <iostream>
#include "ntpoint6d.h"

class TreeAdapter6d
{
private:
	const ntpoint6d* data_;
	size_t size_;
public:
	class SpaceAdapter;
	class VelocityAdapter;

	TreeAdapter6d(const std::vector<ntpoint6d>& vec) :
		data_(vec.data()), size_(vec.size()) { }

	TreeAdapter6d(const ntpoint6d* data, size_t size)
		: data_(data), size_(size) { }

	const ntpoint6d* data() const {
		return data_;
	}

	const size_t size() const {
		return size_;
	}
};

class TreeAdapter6d::SpaceAdapter {
private:
	const ntpoint6d* data_;
	size_t size_;
public:
	typedef ntpoint6d_iterator_r const_iterator;

	TreeAdapter6d::SpaceAdapter() :data_(nullptr), size_(0) { }

	TreeAdapter6d::SpaceAdapter(const TreeAdapter6d& ta) {
		data_ = ta.data();
		size_ = ta.size();
	}

	ntpoint6d_iterator_r begin() const {
		return ntpoint6d_iterator_r(data_);
	}

	ntpoint6d_iterator_r end() const {
		return ntpoint6d_iterator_r(data_ + size_);
	}
};

class TreeAdapter6d::VelocityAdapter {
private:
	const ntpoint6d_iterator_r* iters_;
	size_t size_;
public:
	typedef ntpoint6d_iterator_v const_iterator;

	TreeAdapter6d::VelocityAdapter() : iters_(nullptr), size_(0) { }

	TreeAdapter6d::VelocityAdapter(const std::vector<ntpoint6d_iterator_r>& iters) {
		iters_ = iters.data();
		size_ = iters.size();
	}

	ntpoint6d_iterator_v begin() const {
		return ntpoint6d_iterator_v(iters_);
	}

	ntpoint6d_iterator_v end() const {
		return ntpoint6d_iterator_v(iters_ + size_);
	}
};