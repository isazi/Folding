// Copyright 2012 Alessio Sclocco <a.sclocco@vu.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <exception>
using std::exception;
#include <iomanip>
using std::fixed;
using std::setprecision;
#include <limits>
using std::numeric_limits;
#include <cmath>
#include <ctime>

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


int main(int argc, char *argv[]) {
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
  unsigned int nrDMsPerBlock = 0;
  unsigned int nrPeriodsPerBlock = 0;
  unsigned int nrBinsPerBlock = 0;
  unsigned int nrDMsPerThread = 0;
  unsigned int nrPeriodsPerThread = 0;
  unsigned int nrBinsPerThread = 0;
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
		
    observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    nrDMsPerBlock = args.getSwitchArgument< unsigned int >("-db");
    nrPeriodsPerBlock = args.getSwitchArgument< unsigned int >("-pb");
    nrBinsPerBlock = args.getSwitchArgument< unsigned int >("-bb");
    nrDMsPerThread = args.getSwitchArgument< unsigned int >("-dt");
    nrPeriodsPerThread = args.getSwitchArgument< unsigned int >("-pt");
    nrBinsPerThread = args.getSwitchArgument< unsigned int >("-bt");
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dms"));
		observation.setNrPeriods(args.getSwitchArgument< unsigned int >("-periods"));
		observation.setFirstPeriod(args.getSwitchArgument< unsigned int >("-period_first"));
		observation.setPeriodStep(args.getSwitchArgument< unsigned int >("-period_step"));
		observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));

	} catch ( exception &err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();

	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate memory
	dedispersedData->allocateHostData(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
	foldedData->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
	foldedData->blankHostData();
	readCounterData->allocateHostData(observation.getNrPeriods() * observation.getNrPaddedBins());
	readCounterData->blankHostData();
	writeCounterData->allocateHostData(observation.getNrPeriods() * observation.getNrPaddedBins());
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
	nrSamplesPerBin->setDeviceReadOnly();

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
	} catch ( OpenCLError &err ) {
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
		clFold.setNrDMsPerBlock(nrDMsPerBlock);
		clFold.setNrPeriodsPerBlock(nrPeriodsPerBlock);
		clFold.setNrBinsPerBlock(nrBinsPerBlock);
		clFold.setNrDMsPerThread(nrDMsPerThread);
		clFold.setNrPeriodsPerThread(nrPeriodsPerThread);
		clFold.setNrBinsPerThread(nrBinsPerThread);
		clFold.generateCode();

		dedispersedData->copyHostToDevice();
		clFold(0, dedispersedData, foldedData, readCounterData, writeCounterData);
		foldedData->copyDeviceToHost();
	} catch ( OpenCLError &err ) {
		cerr << err.what() << endl;
		return 1;
	}

	// Check
	CLData< dataType > * CPUFolded = new CLData<dataType >("CPUFolded", true);
	CLData< unsigned int > * CPUCounter = new CLData< unsigned int >("CPUCounter", true);
	CPUFolded->allocateHostData(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
	CPUCounter->allocateHostData(observation.getNrPeriods() * observation.getNrPaddedBins());
	CPUCounter->blankHostData();
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

  if ( wrongValues > 0 ) {
  	cout << "Wrong samples: " << wrongValues << " (" << (wrongValues * 100) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins()) << "%)." << endl;
  } else {
    cout << "TEST PASSED." << endl;
  }

	return 0;
}

