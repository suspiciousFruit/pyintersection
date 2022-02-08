#pragma once
#include <sstream>
#include "../../Intersector/Intersector3d.h"
#include "../../TreeAdapter/TreeAdapter.h"
#include "Utility.h"

// Calculated relatively from project root
static const std::string DATAPATH = "Test\\Datasets\\";
static const std::string RESPATH = "Test\\Results\\";


/*
	Load files from Datasets folder
	Write results to Results folder
*/


#include <iomanip>

template <typename T>
void write_results(const std::vector<collision3d<T>>& colls,
	const std::string& cubefilename, const std::string& pointsfilename)
{
	std::ofstream pointsfile(RESPATH + pointsfilename);
	std::ofstream cubefile(RESPATH + cubefilename);

	pointsfile << "cid,m,n,t,x,y,z\n";
	cubefile << "cid,x_down,x_up,y_down,y_up,z_down,z_up\n";

	size_t cube_index = 0;
	bool manifold_num;

	// Set precision
	pointsfile << std::fixed << std::setprecision(15);

	for (const auto col : colls)
	{
		const auto& c = col.cube;
		cubefile << cube_index << ','
			<< c.x_down << ',' << c.x_up << ','
			<< c.y_down << ',' << c.y_up << ','
			<< c.z_down << ',' << c.z_up << '\n';
		++cube_index;

		manifold_num = 0;
		for (const auto& p : col.apoints)
			pointsfile << cube_index << ','
			<< manifold_num << ','
			<< p << '\n';

		manifold_num = 1;
		for (const auto& p : col.bpoints)
			pointsfile << cube_index << ','
			<< manifold_num << ','
			<< p << '\n';
	}
}


void intersect_and_write_table(const std::string& plane1, const std::string& plane2,
	const std::string& cubefilename, const std::string& pointsfilename, double eps)
{
	const auto apts = read_csv<ntpoint3d>(DATAPATH + plane1);
	const auto bpts = read_csv<ntpoint3d>(DATAPATH + plane2);

	Intersector3d<TreeAdapter<std::vector<ntpoint3d>>> inter;
	const auto colls = inter.intersect(TreeAdapter(apts), TreeAdapter(bpts), eps);

	write_results(colls, cubefilename, pointsfilename);
}

void intersect_and_write_tables(const std::string& plane1, const std::string& plane2,
	const std::string& cubefilename, const std::string& pointsfilename,
	const std::vector<double>& precisions)
{
	const auto apts = read_csv<ntpoint3d>(DATAPATH + plane1);
	const auto bpts = read_csv<ntpoint3d>(DATAPATH + plane2);
	Intersector3d<TreeAdapter<std::vector<ntpoint3d>>> inter;

	for (const auto precision : precisions)
	{
		const auto colls = inter.intersect(TreeAdapter(apts), TreeAdapter(bpts), precision);
		{
			std::ostringstream suffix;
			suffix << std::scientific << "_p" << precision << ".csv";

			write_results(colls,
				cubefilename + suffix.str(),
				pointsfilename + suffix.str());
		}
	}
}

template <typename T>
void write(const std::vector<collision3d<T>>& colls,
	const char* cubefilename, const char* pointsfilename)
{
	std::ofstream pointsfile(pointsfilename);
	std::ofstream cubefile(cubefilename);

	pointsfile << "cid,m,n,t,x,y,z\n";
	cubefile << "cid,x_down,x_up,y_down,y_up,z_down,z_up\n";

	size_t cube_index = 0;
	bool manifold_num;

	// Set precision
	pointsfile << std::fixed << std::setprecision(15);

	for (const auto col : colls)
	{
		const auto& c = col.cube;
		cubefile << cube_index << ','
			<< c.x_down << ',' << c.x_up << ','
			<< c.y_down << ',' << c.y_up << ','
			<< c.z_down << ',' << c.z_up << '\n';
		++cube_index;

		manifold_num = 0;
		for (const auto& p : col.apoints)
			pointsfile << cube_index << ','
			<< manifold_num << ','
			<< p << '\n';

		manifold_num = 1;
		for (const auto& p : col.bpoints)
			pointsfile << cube_index << ','
			<< manifold_num << ','
			<< p << '\n';
	}
}

void __intersect(const char* plane1, const char* plane2,
	const char* cubefilename, const char* pointsfilename, double precision = 0.0001)
{
	const auto apts = read_csv<ntpoint3d>(plane1);
	const auto bpts = read_csv<ntpoint3d>(plane2);
	Intersector3d<TreeAdapter<std::vector<ntpoint3d>>> inter;

	const auto colls = inter.intersect(TreeAdapter(apts), TreeAdapter(bpts), precision);
	write(colls, cubefilename, pointsfilename);
}
