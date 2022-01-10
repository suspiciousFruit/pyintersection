#pragma once
#include "ntpoint3d.h"

class TreeAdapter3d {
private:
		const ntpoint3d* data_;
		size_t size_;
public:
		typedef ntpoint_iterator const_iterator;
		
		TreeAdapter3d() : data_(nullptr), size_(0) { }
		TreeAdapter3d(const ntpoint3d* data, size_t size) : data_(data), size_(size) { }

		ntpoint_iterator begin() const {
			return ntpoint_iterator(data_);
		}
        
		ntpoint_iterator end() const {
			return ntpoint_iterator(data_ + size_);
		}
};