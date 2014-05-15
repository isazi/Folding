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
using PulsarSearch::traditionalFolding;
#include <Bins.hpp>
using PulsarSearch::getNrSamplesPerBin;

typedef float dataType;
const string typeName("float");


int main(int argc, char *argv[]) {
	bool print = false;
	long long unsigned int wrongValues = 0;
	Observation< dataType > observation("FoldingTest", typeName);
	CLData< dataType > * dedispersedData = new CLData< dataType >("DedispersedData", true);
	CLData< dataType > * dedispersedDataTraditional = new CLData< dataType >("DedispersedDataTraditional", true);
	CLData< dataType > * foldedDataCPU = new CLData<dataType >("FoldedDataCPU", true);
	CLData< dataType > * foldedDataTraditional = new CLData<dataType >("FoldedDataTraditional", true);
	CLData< unsigned int > * counterData = new CLData< unsigned int >("CounterData", true);
	CLData< unsigned int > * counterDataTraditional = new CLData< unsigned int >("CounterDataTraditional", true);

	try {
		ArgumentList args(argc, argv);

		print = args.getSwitch("-print");

		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    observation.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
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
  std::vector< std::vector< dataType > * > hostDataBucket(observation.getNrSeconds());
  std::vector< std::vector< dataType > * > hostDataBucketTraditional(observation.getNrSeconds());
  for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
    hostDataBucket[second] = new std::vector< dataType >(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
    hostDataBucketTraditional[second] = new std::vector< dataType >(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond());
  }
	foldedDataCPU->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
	foldedDataCPU->blankHostData();
	foldedDataTraditional->allocateHostData(observation.getNrDMs() * observation.getNrPaddedBins() * observation.getNrPeriods());
	foldedDataTraditional->blankHostData();
	counterData->allocateHostData(observation.getNrPeriods() * observation.getNrPaddedBins());
	counterData->blankHostData();
	counterDataTraditional->allocateHostData(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
	counterDataTraditional->blankHostData();

	srand(time(NULL));
  for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
        (hostDataBucket[second])->at((sample * observation.getNrPaddedDMs()) + DM) = static_cast< dataType >(rand() % 100);
        (hostDataBucketTraditional[second])->at((DM * observation.getNrSamplesPerPaddedSecond()) + sample) = (hostDataBucket[second])->at((sample * observation.getNrPaddedDMs()) + DM);
      }
    }
  }

	// Test & Check
  for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
    folding(0, observation, dedispersedData->getHostData(), foldedDataCPU->getHostData(), counterData->getHostData());
    traditionalFolding(0, observation, dedispersedDataTraditional->getHostData(), foldedDataTraditional->getHostData(), counterDataTraditional->getHostData());
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
    }
	}

  if ( wrongValues > 0 ) {
  	cout << "Wrong samples: " << wrongValues << " (" << (wrongValues * 100) / (static_cast< long long unsigned int >(observation.getNrSeconds()) * observation.getNrDMs() * observation.getNrPeriods() * observation.getNrBins()) << "%)." << endl;
  } else {
    cout << "TEST PASSED." << endl;
  }
  cout << endl;

	return 0;
}

