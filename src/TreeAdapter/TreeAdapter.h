#pragma once
#include "npoint3d.h"
#include "ntpoint3d.h"
#include "ntpoint6d.h"
#include <vector>



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

template <>
class TreeAdapter<std::vector<ntpoint6d>>
{
public:
	class VelocityAdapter
	{
	private:
		const ntpoint6d_iterator_r* data_;
		size_t size_;
	public:
		typedef ntpoint6d_iterator_v const_iterator;

		VelocityAdapter(const std::vector<ntpoint6d_iterator_r>& vec) :
			data_(vec.data()), size_(vec.size())
		{ }

		VelocityAdapter() : data_(nullptr), size_(0)
		{ }

		ntpoint6d_iterator_v begin() const
		{
			return ntpoint6d_iterator_v(data_);
		}

		ntpoint6d_iterator_v end() const
		{
			return ntpoint6d_iterator_v(data_ + size_);
		}
	};

	class SpaceAdapter
	{
	private:
		const ntpoint6d* data_;
		size_t size_;
	public:
		typedef ntpoint6d_iterator_r const_iterator;

		SpaceAdapter(const std::vector<ntpoint6d>& vec) :
			data_(vec.data()), size_(vec.size())
		{ }

		SpaceAdapter() : data_(nullptr), size_(0)
		{ }

		ntpoint6d_iterator_r begin() const
		{
			return ntpoint6d_iterator_r(data_);
		}

		ntpoint6d_iterator_r end() const
		{
			return ntpoint6d_iterator_r(data_ + size_);
		}
	};
};
