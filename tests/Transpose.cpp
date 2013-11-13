/*
 * Copyright (C) 2012
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
#include <ctime>
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
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <CLData.hpp>
#include <utils.hpp>
#include <Transpose.hpp>
using isa::utils::ArgumentList;
using isa::utils::same;
using AstroData::Observation;
using isa::OpenCL::initializeOpenCL;
using isa::OpenCL::CLData;
using PulsarSearch::Transpose;

typedef float dataType;
const string typeName("float");
const unsigned int padding = 32;

// Common parameters
const unsigned int nrBeams = 1;
const unsigned int nrStations = 64;
// LOFAR
//const unsigned int nrSamplesPerSecond = 200000;
// Apertif
const unsigned int nrSamplesPerSecond = 20000;
// DMs
const unsigned int nrDMs = 256;


int main(int argc, char *argv[]) {
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	long long unsigned int wrongValues = 0;
	Observation< dataType > observation("TransposeTest", typeName);
	CLData< dataType > * dedispersedData = new CLData< dataType >("DedispersedData", true);
	CLData< dataType > * transposeData = new CLData< dataType >("TransposeData", true);

	try {
		ArgumentList args(argc, argv);

		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");

	} catch ( exception &err ) {
		cerr << err.what() << endl;
		return 1;
	}
	
	// Setup of the observation
	observation.setPadding(padding);
	observation.setNrSamplesPerSecond(nrSamplesPerSecond);
	observation.setNrDMs(nrDMs);
	
	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();
	
	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate memory
	dedispersedData->allocateHostData(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond());
	transposeData->allocateHostData(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());

	dedispersedData->setCLContext(clContext);
	dedispersedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	transposeData->setCLContext(clContext);
	transposeData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));

	try {
		dedispersedData->allocateDeviceData();
		transposeData->allocateDeviceData();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	srand(time(NULL));
	for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
		for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
			dedispersedData->setHostDataItem((DM * observation.getNrSamplesPerPaddedSecond()) + sample, rand() % 100);
		}
	}

	// Test
	try {
		// Generate kernel
		Transpose< dataType > clTranspose("clTranspose", typeName);
		clTranspose.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
		clTranspose.setObservation(&observation);
		clTranspose.setNrThreadsPerBlock(512);
		clTranspose.setNrDMsPerBlock(32);
		clTranspose.setNrSamplesPerBlock(32);
		clTranspose.generateCode();

		dedispersedData->copyHostToDevice();
		clTranspose(dedispersedData, transposeData);
		transposeData->copyDeviceToHost();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}
	
	// Check
	for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
		for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
			if ( !same(dedispersedData->getHostDataItem((DM * observation.getNrSamplesPerPaddedSecond()) + sample), transposeData->getHostDataItem((sample * observation.getNrPaddedDMs()) + DM)) ) {
				wrongValues++;
			}
		}
	}

	cout << endl;
	cout << "Wrong samples: " << wrongValues << " (" << (wrongValues * 100) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrSamplesPerSecond()) << "%)." << endl;
	cout << endl;

	return 0;
}
