#pragma once
#include <iostream>
#include <arrayobject.h>

#include "../lib/TreeAdapter/ntpoint3d.h"

namespace npApi {

	/// Iterator for numpy point3d array
    class iterator_numpy_ntpoint3d {
    private:
        const ntpoint3d* base_; /// Current point
        const size_t stride_; /// Offset in sizeof(double) bytes
    public:
        iterator_numpy_ntpoint3d(const char* data, size_t stride) :
            base_((const ntpoint3d*)data), stride_(stride / sizeof(double))
		{ }

        inline const point3d& operator* () const {
			return base_->point;
		}

		inline iterator_numpy_ntpoint3d& operator++ () {
			const double* ptr = (const double*)base_;
			ptr += stride_;
			base_ = (const ntpoint3d*)ptr;

			return *this;
		}

		inline bool operator!= (const iterator_numpy_ntpoint3d other) const {
			return base_ != other.base_;
		}

		inline bool operator!= (const ntpoint3d* other) const {
			return base_ != other;
		}

		inline double getnumber() const {
			return base_->number;
		}

		inline double gett() const {
			return base_->t;
		}

		inline const point3d& getpoint() const {
			return base_->point;
		}

		inline const ntpoint3d& ntpoint() const {
			return *base_;
		}

		friend std::ostream& operator<< (std::ostream& stream, const iterator_numpy_ntpoint3d it) {
			const char del = ',';
			stream << it.base_->number << del << it.base_->t << del << *it;
			return stream;
		}
    };

    class TreeAdapterNumpy3d
	{
	private:
		const char* data_;
		size_t size_;
        size_t stride_;
	public:
		typedef iterator_numpy_ntpoint3d const_iterator;

		TreeAdapterNumpy3d() : data_(nullptr), size_(0), stride_(0)
		{ }

		TreeAdapterNumpy3d(PyArrayObject* ndarray) :
            data_((const char*)ndarray->data),
            size_(ndarray->dimensions[0]),
            stride_(ndarray->strides[0]) { }

		iterator_numpy_ntpoint3d begin() const {
			return iterator_numpy_ntpoint3d(data_, stride_);
		}

		iterator_numpy_ntpoint3d end() const {
			return iterator_numpy_ntpoint3d(data_ + size_ * stride_, stride_);
		}
	};
}