#pragma once
/*const size_t N = 4;
struct Point
{
	double data[N];

	inline double& operator[] (size_t i)
	{
		return data[i];
	}

	inline const double& operator[] (size_t i) const
	{
		return data[i];
	}

	Point(const std::initializer_list<double>& list)
	{
		size_t i = 0;
		for (const auto a : list)
		{
			data[i] = a;
			++i;
		}
	}
};

const size_t size = 3;
Point center = { 0, 0, 0 };
double sizes[size] = { 4, 4, 4 };
Point p = { 0, 0, 0 };

#include <functional>
template <size_t N>
void recurse(int* arr, size_t i, void(*f)(int*))
{
	if (i == N)
	{
		f(arr);
		return;
	}

	arr[i] = 1;
	recurse<N>(arr, i + 1, f);
	arr[i] = -1;
	recurse<N>(arr, i + 1, f);
}

template <size_t N>
void make_recurse(void(*f)(int*))
{
	int arr[N];
	recurse<N>(arr, 0, f);
}
#define Size 2
int count;

make_recurse<Size>([](int* arr) {
	static int a = 0;
	for (size_t i = 0; i < Size; ++i)
		p[i] = center[i] + arr[i] * sizes[i] / 2.0;

	for (size_t i = 0; i < Size; ++i)
		std::cout << p[i] << ' ';
	std::cout << '\n';
	++a;
}); */

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////




/*template <typename Container>
class Trajectory
{
private:
	Container array_;
public:
	static Trajectory from_file(const char* filename)
	{
		Container container;
		decltype(__fromref(*container.begin())) p;
		std::ifstream file(filename);

		while (1)
		{
			file >> p;
			container.push_back(p);
			if (file.eof()) break;
		}

		return container;
	}
	point* data()
	{
		return array_.data();
	}
	size_t size()
	{
		return array_.size();
	}
	template <typename Container>
	void get_points(const point* a, const point* b, Container&);
};*/



/////////////////////////////////////////////////////////////////////////////////////
//
//
//
//
//
//
/////////////////////////////////////////////////////////////////////////////////////


//template <typename T>
//T _get(const T& t)
//{
//	return t;
//}
//
//
//template <typename ContT>
//auto _getconttype()
//{
//	ContT c;
//	return __get(*(c.begin()));
//}
//
//#include <vector>
//class IPoint
//{
//public:
//	class ICube
//	{
//	public:
//		bool isfit(const IPoint*) const;
//	};
//};
//
//
//
//template <typename PointT, typename ContT = std::vector<PointT>>
//class Tree
//{
//public:
//	void update_cube(const typename PointT::Cube&);
//
//	template <typename Container>
//	void sieve_base(const Container&);
//	template <typename Container>
//	void sieve_other(const Container&);
//
//	template <typename First, typename Last>
//	void sieve_base(First, Last);
//	template <typename First, typename Last>
//	void sieve_other(First, Last);
//
//	void clean_buffers();
//
//	template <typename Container>
//	void get_full_buffers(Container&);
//};

struct cube3d { };

template <typename NextL>
class InputLayer
{
private:
	cube3d cube_;
	NextL* childs_[8];
public:

};

class MiddleLayer
{
private:
	cube3d cube_;
	MiddleLayer* childs_[8];
public:
	void sievea();
	void sieveb();
};

class PreEndLayer
{
private:
	cube3d cube_;
	EndLayer* childs_[8];
public:
	void sievea();
	void sieveb();
};

class array { };
class EndLayer
{
private:
	cube3d cube_;
	array base_;
	array other_;
};

void builder()
{

}



#include <cmath>
//namespace LineTree {
//
//	struct Leaf
//	{
//		static const size_t NDIM = 8;
//		const cube3d* base_layer_pointeer;
//		const cube3d* base_leaf_pointeer;
//		size_t depth;
//
//		void next(int fitnumber)
//		{
//			base_layer_pointeer = base_layer_pointeer + size_t(std::pow(2, depth));
//			base_leaf_pointeer = base_layer_pointeer + NDIM * fitnumber;
//			++depth;
//		}
//
//		int sieve(const int*)
//		{
//			for (size_t i = 0; i < NDIM; ++i)
//
//		}
//	};
//
//	constexpr size_t pow(const size_t a, const size_t b)
//	{
//		size_t res = 1;
//		for (size_t i = 0; i < b; ++i)
//			res *= a;
//		return res;
//	};
//
//	constexpr size_t DIM(size_t d)
//	{
//		size_t res = 0;
//		for (size_t i = 0; i < d; ++i)
//			res += size_t(pow(2, i));
//
//		return res;
//	}
//
//	class Tree
//	{
//	private:
//		cube3d data[DIM(3)];
//		const size_t NDIM = 8;
//		const size_t DEPTH = 3;
//		static int getfitnumber(const Leaf& leaf)
//		{
//
//		}
//
//	public:
//		void sievea()
//		{
//			Leaf leaf;
//
//			for (size_t i = 0; i < DEPTH; ++i)
//			{
//				int fitnumber = getfitnumber(leaf);
//				if (fitnumber == -1) break;
//				leaf.next(fitnumber);
//			}
//		}
//	};
//}
