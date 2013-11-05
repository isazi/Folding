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
#include <Folding.hpp>
#include <FoldingCPU.hpp>
using isa::utils::ArgumentList;
using isa::utils::same;
using AstroData::Observation;
using isa::OpenCL::initializeOpenCL;
using isa::OpenCL::CLData;
using PulsarSearch::folding;
using PulsarSearch::Folding;

typedef float dataType;
const string typeName("float");
const unsigned int padding = 32;

// Common parameters
const unsigned int nrBeams = 1;
const unsigned int nrStations = 64;
// LOFAR
/*const float minFreq = 138.965f;
const float channelBandwidth = 0.195f;
const unsigned int nrSamplesPerSecond = 200000;
const unsigned int nrChannels = 32;*/
// Apertif
const float minFreq = 1425.0f;
const float channelBandwidth = 0.2929f;
const unsigned int nrSamplesPerSecond = 20000;
const unsigned int nrChannels = 1024;
// DMs
const unsigned int nrDMs = 256;
// Periods
const unsigned int nrPeriods = 128;
const unsigned int nrBins = 256;


int main(int argc, char *argv[]) {
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	long long unsigned int wrongValues = 0;
	Observation< dataType > observation("FoldingTest", typeName);
	CLData< dataType > * dedispersedData = new CLData< dataType >("DedispersedData", true);
	CLData< dataType > * foldedData = new CLData<dataType >("FoldedData", true);
	CLData< unsigned int > * counterData = new CLData< unsigned int >("CounterData", true);

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
	observation.setNrPeriods(nrPeriods);
	observation.setFirstPeriod(nrBins);
	observation.setPeriodStep(nrBins);
	observation.setNrBins(nrBins);
	
	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();
	
	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate memory
	dedispersedData->allocateHostData(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
	foldedData->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
	foldedData->blankHostData();
	counterData->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
	counterData->blankHostData();

	dedispersedData->setCLContext(clContext);
	dedispersedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	foldedData->setCLContext(clContext);
	foldedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	counterData->setCLContext(clContext);
	counterData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));

	try {
		dedispersedData->allocateDeviceData();
		foldedData->allocateDeviceData();
		foldedData->copyHostToDevice();
		counterData->allocateDeviceData();
		counterData->copyHostToDevice();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	srand(time(NULL));
	for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
		for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
			dedispersedData->setHostDataItem((sample * observation.getNrPaddedDMs()) + DM, rand() % 100);
		}
	}

	// Test
	try {
		// Generate kernel
		Folding< dataType > clFold("clFold", typeName);
		clFold.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
		clFold.setObservation(&observation);
		clFold.setNrDMsPerBlock(128);
		clFold.setNrPeriodsPerBlock(2);
		clFold.setNrBinsPerBlock(1);
		clFold.setNrDMsPerThread(2);
		clFold.setNrPeriodsPerThread(2);
		clFold.setNrBinsPerThread(4);
		clFold.generateCode();

		dedispersedData->copyHostToDevice();
		clFold(dedispersedData, foldedData, counterData);
		foldedData->copyDeviceToHost();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}
	
	// Check
	CLData< dataType > * CPUFolded = new CLData<dataType >("CPUFolded", true);
	CLData< unsigned int > * CPUCounter = new CLData< unsigned int >("CPUCounter", true);
	folding(0, observation, dedispersedData->getHostData(), CPUFolded->getHostData(), CPUCounter->getHostData());
	for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
		for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
			for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
				const unsigned int dataItem = (bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + DM;
				if ( !same(CPUFolded->getHostDataItem(dataItem), foldedData->getHostDataItem(dataItem)) ) {
					wrongValues++;
				}
			}
		}
	}	

	cout << "Wrong samples: " << wrongValues << " (" << (wrongValues * 100) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins()) << "%)." << endl;
	cout << endl;

	return 0;
}

