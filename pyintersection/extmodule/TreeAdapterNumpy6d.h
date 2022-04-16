#pragma once
#include <iostream>
#include <arrayobject.h>

#include "../lib/TreeAdapter/ntpoint3d.h"

// TODO remake strange structure of subclasses SpaceAdapter and VelocityAdapter

class TreeAdapterNumpy6d {
private:
	const char* data_;
	size_t size_;
    size_t stride_; // bytes stride
public:
	class SpaceAdapter;
	class VelocityAdapter;

    TreeAdapterNumpy6d(PyArrayObject* ndarray) :
        data_((const char*)ndarray->data),
        size_(ndarray->dimensions[0]),
        stride_(ndarray->strides[0]) { }

	// TreeAdapter6d(const std::vector<ntpoint6d>& vec) :
	// 	data_(vec.data()), size_(vec.size()) { }

	// TreeAdapter6d(const ntpoint6d* data, size_t size)
	// 	: data_(data), size_(size) { }

	const char* data() const {
		return data_;
	}

	size_t size() const {
		return size_;
	}

    size_t stride() const {
        return stride_;
    }
};

class iterator_numpy_ntpoint6d_v;

// Iterator for space dimention
class iterator_numpy_ntpoint6d_r {
private:
	friend class iterator_numpy_ntpoint6d_v;
	const ntpoint6d* base_;
    const size_t stride_; /// Offset in sizeof(double) bytes
public:
    iterator_numpy_ntpoint6d_r(const char* data, size_t stride) :
        base_((const ntpoint6d*)data), stride_(stride / sizeof(double))
    { }

	const point3d& operator* () const {
		return base_->point.r;
	}

	iterator_numpy_ntpoint6d_r& operator++ () {
		const double* ptr = (const double*)base_;
		ptr += stride_;
		base_ = (const ntpoint6d*)ptr;

		return *this;
	}

	bool operator!= (const iterator_numpy_ntpoint6d_r other) {
		return base_ != other.base_;
	}

	const ntpoint6d& ntpoint() const {
		return *base_;
	}

	friend std::ostream& operator<< (std::ostream& s, const iterator_numpy_ntpoint6d_r it) {
		s << *(it.base_);
		return s;
	}
};


// Iterator for velocity dimention based on space iterator
class iterator_numpy_ntpoint6d_v {
private:
	const iterator_numpy_ntpoint6d_r* iterator_r_;
public:
	iterator_numpy_ntpoint6d_v(const iterator_numpy_ntpoint6d_r* base) :
		iterator_r_(base)
	{ }

	const point3d& operator* () const {
		return iterator_r_->base_->point.v;
	}

	iterator_numpy_ntpoint6d_v& operator++ () {
		++iterator_r_;
		return *this;
	}

	bool operator!= (const iterator_numpy_ntpoint6d_v other) {
		return iterator_r_ != other.iterator_r_;
	}

	friend std::ostream& operator<< (std::ostream& s, iterator_numpy_ntpoint6d_v p) {
		s << *(p.iterator_r_);
		return s;
	}

	const ntpoint6d& ntpoint() const {
		return *iterator_r_->base_;
	}
};

class TreeAdapterNumpy6d::SpaceAdapter {
private:
	const char* data_;
	size_t size_;
    size_t stride_;
public:
	typedef iterator_numpy_ntpoint6d_r const_iterator;

	TreeAdapterNumpy6d::SpaceAdapter() :
        data_(nullptr), size_(0), stride_(0) { }

	TreeAdapterNumpy6d::SpaceAdapter(const TreeAdapterNumpy6d& ta) {
		data_ = ta.data();
		size_ = ta.size();
        stride_ = ta.stride();
	}

	iterator_numpy_ntpoint6d_r begin() const {
		return iterator_numpy_ntpoint6d_r(data_, stride_);
	}

	iterator_numpy_ntpoint6d_r end() const { // Can skip last parameter
		return iterator_numpy_ntpoint6d_r(data_ + size_ * stride_, stride_);
	}
};

class TreeAdapterNumpy6d::VelocityAdapter {
private:
	const iterator_numpy_ntpoint6d_r* iters_;
	size_t size_;
public:
	typedef iterator_numpy_ntpoint6d_v const_iterator;

	TreeAdapterNumpy6d::VelocityAdapter() :
        iters_(nullptr), size_(0) { }

	TreeAdapterNumpy6d::VelocityAdapter(const std::vector<iterator_numpy_ntpoint6d_r>& iters) {
		iters_ = iters.data();
		size_ = iters.size();
	}

	iterator_numpy_ntpoint6d_v begin() const {
		return iterator_numpy_ntpoint6d_v(iters_);
	}

	iterator_numpy_ntpoint6d_v end() const {
		return iterator_numpy_ntpoint6d_v(iters_ + size_);
	}
};
