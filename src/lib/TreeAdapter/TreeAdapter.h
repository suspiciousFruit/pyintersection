#pragma once
#include "npoint3d.h"
#include "ntpoint3d.h"
 #include "ntpoint6d.h"
#include <vector>

#include "TreeAdapter3d.h"
#include "TreeAdapter6d.h"



template <typename T>
class TreeAdapter
{
public:
	TreeAdapter(const T&);
};


template <>
class TreeAdapter<std::vector<npoint3d>>
{
private:
	const npoint3d* data_;
	size_t size_;
public:
	typedef npoint_iterator const_iterator;

	TreeAdapter() : data_(nullptr), size_(0)
	{ }

	TreeAdapter(const std::vector<npoint3d>& vec) :
		data_(vec.data()), size_(vec.size())
	{ }

	npoint_iterator begin() const
	{
		return npoint_iterator(data_);
	}

	npoint_iterator end() const
	{
		return npoint_iterator(data_ + size_);
	}
};

template <>
class TreeAdapter<std::vector<ntpoint3d>>
{
private:
	const ntpoint3d* data_;
	size_t size_;
public:
	typedef ntpoint_iterator const_iterator;

	TreeAdapter() : data_(nullptr), size_(0)
	{ }

	TreeAdapter(const std::vector<ntpoint3d>& vec) :
		data_(vec.data()), size_(vec.size())
	{ }

	ntpoint_iterator begin() const
	{
		return ntpoint_iterator(data_);
	}

	ntpoint_iterator end() const
	{
		return ntpoint_iterator(data_ + size_);
	}
};

typedef TreeAdapter<std::vector<npoint3d>> VectorAdapter;
