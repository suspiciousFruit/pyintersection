#pragma once
#include <vector>
#include "MiddleLeaf.h"
#include "LastLeaf.h"
#include "./collision3d.h"

/*
	����� ������, ������� ���������� ����� � ��������� � ����� ���������� �������
	����� ����������� � �����, ������� ������ � ��� �����

	IIterator ��� ����� ������� ����� ��������� ������ � ����� �����������
	��� ������������� ������ ���������� const point3d&
*/


template <typename IIterator>
class Tree
{
private:
	Leaf<IIterator>* root_;
	size_t depth_;
	std::vector<LastLeaf<IIterator>*> lasts_;
public:
	Tree(const cube3d& space, size_t depth);
	Tree(size_t depth);

	~Tree();

	void update_cube(const cube3d&);
	const cube3d& get_cube() const;

	size_t get_depth() const;

	void clean_buffers();
	template <typename Container>
	void get_full_buffers(Container& container);

	/*
		������� ������� ���������� ����� �� ����������� �� ����������
		��������� ������ ���� ����������� �� ���� �������� STL
		� ������� ������������ ������������� const_iterator
	*/
	template <typename Container>
	void sieve_a(const Container&);
	template <typename Container>
	void sieve_b(const Container&);

	template <typename Container>
	void sieve_a_iterable(const Container&);
	template <typename Container>
	void sieve_b_iterable(const Container&);


	// Debug functions
	void print_buffers()
	{
		for (const auto last : lasts_)
			last->print();
	}
};



template <typename IIterator>
Tree<IIterator>::Tree(const cube3d& cube, size_t depth) : depth_(depth)
{
	if (depth == 0) throw std::exception("Depth can't be 0!");

	root_ = new MiddleLeaf<IIterator>(cube);
	root_->make_childs(depth);
	root_->check_in_if_last(lasts_);
}

template <typename IIterator>
Tree<IIterator>::Tree(size_t depth) : depth_(depth)
{
	//if (depth == 0) throw std::exception("Depth can't be 0!");

	root_ = new MiddleLeaf<IIterator>();
	root_->make_childs(depth);
	root_->check_in_if_last(lasts_);
}

template <typename IIterator>
Tree<IIterator>::~Tree()
{
	delete root_;
}

template <typename IIterator>
template <typename Container>
void Tree<IIterator>::sieve_a(const Container& array)
{
	for (auto iter = std::begin(array); iter != std::end(array); ++iter)
		root_->sieve_a(iter);
}

template <typename IIterator>
template <typename Container>
void Tree<IIterator>::sieve_b(const Container& array)
{
	for (auto iter = std::begin(array); iter != std::end(array); ++iter)
		root_->sieve_b(iter);
}

template <typename IIterator>
template <typename Container>
void Tree<IIterator>::sieve_a_iterable(const Container& array)
{
	for (auto iter = std::begin(array); iter != std::end(array); ++iter)
		root_->sieve_a(*iter);
}

template <typename IIterator>
template <typename Container>
void Tree<IIterator>::sieve_b_iterable(const Container& array)
{
	for (auto iter = std::begin(array); iter != std::end(array); ++iter)
		root_->sieve_b(*iter);
}

template <typename IIterator>
void Tree<IIterator>::update_cube(const cube3d& cube)
{
	root_->update_cube(cube);
}

template <typename IIterator>
const cube3d& Tree<IIterator>::get_cube() const
{
	return root_->get_cube();
}

template <typename IIterator>
size_t Tree<IIterator>::get_depth() const
{
	return depth_;
}

template <typename IIterator>
void Tree<IIterator>::clean_buffers()
{
	for (const auto last : lasts_)
		last->clear_arrays();
}

template <typename IIterator>
template <typename Container>
void Tree<IIterator>::get_full_buffers(Container& cont)
{
	for (auto last : lasts_)
		if (last->has_both_points())
		{
			auto&& apoints = last->get_apoints();
			auto&& bpoints = last->get_bpoints();
			auto& cube = last->get_cube();

			const collision3d<IIterator> collision(
				std::move(apoints), 
				std::move(bpoints), cube);
			cont.push_back(collision);
		}
}

