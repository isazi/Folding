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
using isa::utils::ArgumentList;
#include <Observation.hpp>
using AstroData::Observation;
#include <InitializeOpenCL.hpp>
using isa::OpenCL::initializeOpenCL;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <utils.hpp>
using isa::utils::same;
#include <Folding.hpp>
using PulsarSearch::Folding;
#include <FoldingCPU.hpp>
using PulsarSearch::folding;
#include <Bins.hpp>
using PulsarSearch::getNrSamplesPerBin;

typedef float dataType;
const string typeName("float");
const unsigned int padding = 32;

// LOFAR
//const unsigned int nrSamplesPerSecond = 200000;
// Apertif
const unsigned int nrSamplesPerSecond = 20000;
// DMs
const unsigned int nrDMs = 256;
// Periods
const unsigned int nrPeriods = 128;
const unsigned int nrBins = 256;
const unsigned int periodStep = 64;


int main(int argc, char *argv[]) {
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	long long unsigned int wrongValues = 0;
	Observation< dataType > observation("FoldingTest", typeName);
	CLData< dataType > * dedispersedData = new CLData< dataType >("DedispersedData", true);
	CLData< dataType > * foldedData = new CLData<dataType >("FoldedData", true);
	CLData< unsigned int > * readCounterData = new CLData< unsigned int >("ReadCounterData", true);
	CLData< unsigned int > * writeCounterData = new CLData< unsigned int >("WriteCounterData", true);
	CLData< unsigned int > * nrSamplesPerBin = new CLData< unsigned int >("NrSamplesPerBin", true);

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
	observation.setPeriodStep(periodStep);
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
	readCounterData->allocateHostData(observation.getNrPeriods() * 2 * observation.getNrPaddedBins());
	readCounterData->blankHostData();
	writeCounterData->allocateHostData(observation.getNrPeriods() * 2 * observation.getNrPaddedBins());
	writeCounterData->blankHostData();
	vector< unsigned int > * nrSamplesPerBinData = getNrSamplesPerBin(observation);
	nrSamplesPerBin->allocateHostData(*nrSamplesPerBinData);

	dedispersedData->setCLContext(clContext);
	dedispersedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	foldedData->setCLContext(clContext);
	foldedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	readCounterData->setCLContext(clContext);
	readCounterData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	writeCounterData->setCLContext(clContext);
	writeCounterData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	nrSamplesPerBin->setCLContext(clContext);
	nrSamplesPerBin->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));

	try {
		dedispersedData->allocateDeviceData();
		foldedData->allocateDeviceData();
		foldedData->copyHostToDevice();
		readCounterData->allocateDeviceData();
		readCounterData->copyHostToDevice();
		writeCounterData->allocateDeviceData();
		writeCounterData->copyHostToDevice();
		nrSamplesPerBin->allocateDeviceData();
		nrSamplesPerBin->copyHostToDevice();
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
		clFold.setNrSamplesPerBin(nrSamplesPerBin);
		clFold.setNrDMsPerBlock(128);
		clFold.setNrPeriodsPerBlock(2);
		clFold.setNrBinsPerBlock(1);
		clFold.setNrDMsPerThread(2);
		clFold.setNrPeriodsPerThread(2);
		clFold.setNrBinsPerThread(4);
		clFold.generateCode();

		dedispersedData->copyHostToDevice();
		clFold(0, dedispersedData, foldedData, readCounterData, writeCounterData);
		foldedData->copyDeviceToHost();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}
	
	// Check
	CLData< dataType > * CPUFolded = new CLData<dataType >("CPUFolded", true);
	CLData< unsigned int > * CPUCounter = new CLData< unsigned int >("CPUCounter", true);
	CPUFolded->allocateHostData(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
	CPUCounter->allocateHostData(observation.getNrPeriods() * 2 * observation.getNrPaddedBins());
	CPUCounter->blankHostData();
	folding(0, observation, dedispersedData->getHostData(), CPUFolded->getHostData(), CPUCounter->getHostData());
	for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
		long long unsigned int wrongValuesBin = 0;

		for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
			for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
				const unsigned int dataItem = (bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + DM;
				if ( !same(CPUFolded->getHostDataItem(dataItem), foldedData->getHostDataItem(dataItem)) ) {
					wrongValues++;
					wrongValuesBin++;
				}
			}
		}

		if ( wrongValuesBin > 0 ) {
			cout << "Wrong samples bin " << bin << ": " << wrongValuesBin << " (" << (wrongValuesBin * 100) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods()) << "%)." << endl;
		}
	}	

	cout << endl;
	cout << "Wrong samples: " << wrongValues << " (" << (wrongValues * 100) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins()) << "%)." << endl;
	cout << endl;

	return 0;
}

