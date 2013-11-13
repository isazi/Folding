/*
 * Copyright (C) 2013
 * Alessio Sclocco <a.sclocco@vu.nl>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::exception;
using std::ofstream;
using std::fixed;
using std::setprecision;
using std::numeric_limits;

#include <ArgumentList.hpp>
using isa::utils::ArgumentList;
#include <Observation.hpp>
using AstroData::Observation;
#include <InitializeOpenCL.hpp>
using isa::OpenCL::initializeOpenCL;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <utils.hpp>
using isa::utils::toStringValue;
#include <Timer.hpp>
using isa::utils::Timer;
#include <Transpose.hpp>
using PulsarSearch::Transpose;

typedef float dataType;
const string typeName("float");
const unsigned int maxThreadsPerBlock = 1024;
const unsigned int padding = 32;


int main(int argc, char * argv[]) {
	unsigned int lowerNrThreads = 0;
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	Observation< dataType > observation("TransposeTuning", typeName);
	CLData< dataType > * dedispersedData = new CLData< dataType >("DedispersedData", true);
	CLData< dataType > * transposedData = new CLData<dataType >("TranposedData", true);


	try {
		ArgumentList args(argc, argv);

		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dms"));
		lowerNrThreads = args.getSwitchArgument< unsigned int >("-lnt");
		observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
	} catch ( exception & err ) {
		cerr << err.what() << endl;
		return 1;
	}

	// Setup of the observation
	observation.setPadding(padding);
	
	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();
	
	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	cout << fixed << endl;
	cout << "# nrDMs nrSamplesPerSecond nrThreadPerBlock GB/s err time err" << endl << endl;

	// Allocate memory
	dedispersedData->allocateHostData(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond());
	dedispersedData->blankHostData();
	transposedData->allocateHostData(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
	transposedData->blankHostData();

	dedispersedData->setCLContext(clContext);
	dedispersedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	transposedData->setCLContext(clContext);
	transposedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));

	try {
		dedispersedData->allocateDeviceData();
		dedispersedData->copyHostToDevice();
		transposedData->allocateDeviceData();
		transposedData->copyHostToDevice();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
	}

	// Find the parameters
	vector< unsigned int > configurations;
	for ( unsigned int threadsPerBlock = lowerNrThreads; threadsPerBlock <= maxThreadsPerBlock; threadsPerBlock++ ) {
		if ( observation.getNrDMs() % threadsPerBlock == 0 ) {
			configurations.push_back(threadsPerBlock);
		}
	}

	for ( vector< unsigned int >::const_iterator configuration = configurations.begin(); configuration != configurations.end(); configuration++ ) {
		double Acur = 0.0;
		double Aold = 0.0;
		double Vcur = 0.0;
		double Vold = 0.0;

		try {
			// Generate kernel
			Transpose< dataType > clTranspose("clTranspose", typeName);
			clTranspose.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
			clTranspose.setObservation(&observation);
			clTranspose.setNrThreadsPerBlock(*configuration);
			clTranspose.generateCode();

			clTranspose(dedispersedData, transposedData);
			(clTranspose.getTimer()).reset();
			
			for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
				clTranspose(dedispersedData, transposedData);
				
				if ( iteration == 0 ) {
					Acur = clTranspose.getGB() / clTranspose.getTimer().getLastRunTime();
				} else {
					Aold = Acur;
					Vold = Vcur;

					Acur = Aold + (((clTranspose.getGB() / clTranspose.getTimer().getLastRunTime()) - Aold) / (iteration + 1));
					Vcur = Vold + (((clTranspose.getGB() / clTranspose.getTimer().getLastRunTime()) - Aold) * ((clTranspose.getGB() / clTranspose.getTimer().getLastRunTime()) - Acur));
				}
			}
			Vcur = sqrt(Vcur / nrIterations);

			cout << observation.getNrDMs() << " " << observation.getNrSamplesPerSecond() << " " << *configuration << " " << setprecision(3) << Acur << " " << Vcur << " " << setprecision(6) << clTranspose.getTimer().getAverageTime() << " " << clTranspose.getTimer().getStdDev() << endl;
		} catch ( OpenCLError err ) {
			cerr << err.what() << endl;
			continue;
		}
	}

	cout << endl;

	return 0;
}
