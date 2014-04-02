// Copyright 2013 Alessio Sclocco <a.sclocco@vu.nl>
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
#include <FoldingAVX.hpp>
using PulsarSearch::folding;
#include <Bins.hpp>
using PulsarSearch::getNrSamplesPerBin;

typedef float dataType;
const string typeName("float");
const unsigned int padding = 8;

// LOFAR
//const unsigned int nrSamplesPerSecond = 200000;
// Apertif
const unsigned int nrSamplesPerSecond = 20000;
// DMs
const unsigned int nrDMs = 2048;
// Periods
const unsigned int nrPeriods = 512;
const unsigned int nrBins = 128;
const unsigned int periodStep = 32;


int main(int argc, char *argv[]) {
	long long unsigned int wrongValues = 0;
	Observation< dataType > observation("FoldingTest", typeName);
	CLData< dataType > * dedispersedData = new CLData< dataType >("DedispersedData", true);
	CLData< dataType > * foldedDataCPU = new CLData<dataType >("FoldedDataCPU", true);
	CLData< dataType > * foldedDataAVX = new CLData<dataType >("FoldedDataAVX", true);
	CLData< unsigned int > * readCounterData = new CLData< unsigned int >("ReadCounterData", true);
	CLData< unsigned int > * writeCounterData = new CLData< unsigned int >("WriteCounterData", true);
	CLData< unsigned int > * nrSamplesPerBin = new CLData< unsigned int >("NrSamplesPerBin", true);

	// Setup of the observation
	observation.setPadding(padding);
	observation.setNrSamplesPerSecond(nrSamplesPerSecond);
	observation.setNrDMs(nrDMs);
	observation.setNrPeriods(nrPeriods);
	observation.setFirstPeriod(nrBins);
	observation.setPeriodStep(periodStep);
	observation.setNrBins(nrBins);


	// Allocate memory
	dedispersedData->allocateHostData(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
	foldedDataCPU->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
	foldedDataCPU->blankHostData();
	foldedDataAVX->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
	foldedDataAVX->blankHostData();
	readCounterData->allocateHostData(observation.getNrPeriods() * 2 * observation.getNrPaddedBins());
	readCounterData->blankHostData();
	writeCounterData->allocateHostData(observation.getNrPeriods() * 2 * observation.getNrPaddedBins());
	writeCounterData->blankHostData();
	vector< unsigned int > * nrSamplesPerBinData = getNrSamplesPerBin(observation);
	nrSamplesPerBin->allocateHostData(*nrSamplesPerBinData);

	srand(time(NULL));
	for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
		for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
			dedispersedData->setHostDataItem((sample * observation.getNrPaddedDMs()) + DM, rand() % 100);
		}
	}

	// Test & Check
	folding(0, observation, dedispersedData->getHostData(), foldedDataCPU->getHostData(), writeCounterData->getHostData());
	folding(0, observation, dedispersedData->getHostData(), foldedDataAVX->getHostData(), readCounterData->getHostData(), writeCounterData->getHostData());
	for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
		long long unsigned int wrongValuesBin = 0;

		for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
			for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
				const unsigned int dataItem = (bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + DM;
				if ( !same(foldedDataCPU->getHostDataItem(dataItem), foldedDataAVX->getHostDataItem(dataItem)) ) {
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

