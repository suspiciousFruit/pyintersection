#pragma once
#include "Intersector/cube6d.h"
#include "Intersector/Tree/collision3d.h"
#include <vector>


//
// Интерфейс для массива точек
// Обязан содержать два метода, которые возвращают итератор на начало и конец массива
//
template <typename Iter>
class IArray
{
public:
	Iter begin() const;
	Iter end() const;
};

// PointArrayT massive with points: iteratble, *iter return const point3d&

//
// Интерфейс для итератора, который возвращает IArray
//
class IIterator
{
private:

public:
	const point3d& operator*() const;
	const point3d* operator-> () const;
	IIterator& operator++ ();
	bool operator= (const IIterator&) const;
};



class ICube
{
public:
	template <typename IIterator>
	bool isfit(const IIterator) const;
};



class Intersector3d
{
public:
	typedef std::vector<cube3d> Cubes3d;
	typedef std::vector<collision3d> Collisions3d;

	Intersector3d(size_t treedepth);

	template <typename IArray>
	Cubes3d findcubes3d(const IArray&, const IArray&,
		size_t niteration);

	template <typename IArray>
	Collisions3d findcollisions3d(const IArray&, const IArray&,
		size_t niteration);
};

class collision6d;
class Intersector6d
{
public:
	typedef std::vector<cube6d> Cubes6d;
	typedef std::vector<collision6d> Collisions6d;

	template <typename IArray>
	Cubes6d findcubes3d(const IArray&, const IArray&,
		size_t niteration);

	template <typename IArray>
	Collisions6d findcollisions3d(const IArray&, const IArray&,
		size_t niteration);
};


//
//
//
class ITree
{
public:
	template <typename _IArray>
	void sievea(const _IArray&);

	template <typename _IArray>
	void sieveb(const _IArray&);

	void updatecube(const cube3d&);

	template <typename OutCont>
	void getcubes(OutCont&);
};



