//
// Copyright (C) 2013
// Alessio Sclocco <a.sclocco@vu.nl>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

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
using PulsarSearch::traditionalFolding;
#include <Bins.hpp>
using PulsarSearch::getNrSamplesPerBin;

typedef float dataType;
const string typeName("float");
const unsigned int padding = 8;

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
	CLData< dataType > * dedispersedDataTraditional = new CLData< dataType >("DedispersedDataTraditional", true);
	CLData< dataType > * foldedDataCPU = new CLData<dataType >("FoldedDataCPU", true);
	CLData< dataType > * foldedDataTraditional = new CLData<dataType >("FoldedDataTraditional", true);
	CLData< unsigned int > * counterData = new CLData< unsigned int >("CounterData", true);
	CLData< unsigned int > * nrSamplesPerBin = new CLData< unsigned int >("NrSamplesPerBin", true);

	try {
		ArgumentList args(argc, argv);
		
		observation.setPadding(padding);
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

	// Allocate memory
	dedispersedData->allocateHostData(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
	dedispersedDataTraditional->allocateHostData(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond());
	foldedDataCPU->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
	foldedDataCPU->blankHostData();
	foldedDataTraditional->allocateHostData(observation.getNrDMs() * observation.getNrPaddedBins() * observation.getNrPeriods());
	foldedDataTraditional->blankHostData();
	counterData->allocateHostData(observation.getNrPeriods() * observation.getNrPaddedBins());
	vector< unsigned int > * nrSamplesPerBinData = getNrSamplesPerBin(observation);
	nrSamplesPerBin->allocateHostData(*nrSamplesPerBinData);

	srand(time(NULL));
	for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
		for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
			dedispersedData->setHostDataItem((sample * observation.getNrPaddedDMs()) + DM, rand() % 100);
			dedispersedDataTraditional->setHostDataItem((DM * observation.getNrSamplesPerPaddedSecond()) + sample, dedispersedData->getHostDataItem((sample * observation.getNrPaddedDMs()) + DM));
		}
	}

	// Test & Check
	counterData->blankHostData();
	folding(0, observation, dedispersedData->getHostData(), foldedDataCPU->getHostData(), counterData->getHostData());
	counterData->blankHostData();
	traditionalFolding(0, observation, dedispersedDataTraditional->getHostData(), foldedDataTraditional->getHostData(), counterData->getHostData());
	for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
		long long unsigned int wrongValuesBin = 0;

		for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
			for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
				if ( !same(foldedDataCPU->getHostDataItem((bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + DM), foldedDataTraditional->getHostDataItem((((DM * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin)) ) {
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

