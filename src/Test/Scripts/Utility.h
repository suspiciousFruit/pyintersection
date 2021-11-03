#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>



template <typename PointT>
auto readpts(const std::string& filename)
{
	std::vector<PointT> res;
	std::ifstream file(filename);

	if (!file.is_open())
	{
		std::cout << "Error: Can not open file" << std::endl;
		return res;
	}

	PointT p;
	while (1)
	{
		file >> p;
		res.emplace_back(p);

		if (file.eof() || !file.good())
			break;
	}

	return res;
}

// Read file and return vector<PointT>
template <typename PointT>
std::vector<PointT> read_csv(const std::string& filename)
{
	std::ifstream file(filename);
	std::vector<PointT> res;

	if (!file.is_open())
	{
		std::cout << "Error: Can not open file" << std::endl;
		return res;
	}

	char symbol;
	while (1)
	{
		symbol = file.get();
		if (symbol == '\n' || !file.good())
			break;
	}

	PointT p;
	while (1)
	{
		file >> p;

		if (file.eof() || !file.good())
			break;

		res.emplace_back(p);
	}

	return res;
}

template <typename T>
void print(const T& C)
{
	for (const auto& c : C)
		std::cout << c << std::endl;
}



void wirte_csv(const char* filename, const std::vector<point3d>& points, const std::vector<const char*>& labels)
{
	std::ofstream file(filename);

	for (size_t i = 0; i < labels.size(); ++i)
		file << labels[i] << (i == labels.size() - 1 ? "\n" : ",");

	for (size_t i = 0; i < points.size(); ++i)
		file << points[i] << '\n';
}
