// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
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
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ctime>

#include <ArgumentList.hpp>
#include <Exceptions.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <Shifts.hpp>
#include <Folding.hpp>


int main(int argc, char *argv[]) {
  bool avx = false;
  bool phi = false;
  unsigned int nrDMsPerThread = 0;
  unsigned int nrPeriodsPerThread = 0;
  unsigned int nrBinsPerThread = 0;
	long long unsigned int wrongSamples = 0;
	AstroData::Observation< float > observation("FoldingTest", "float");

	try {
    isa::utils::ArgumentList args(argc, argv);
    avx = args.getSwitch("-avx");
    phi = args.getSwitch("-phi");
    if ( ! (avx || phi) ) {
      throw isa::Exceptions::EmptyCommandLine();
    }
    observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    nrDMsPerThread = args.getSwitchArgument< unsigned int >("-dt");
    nrPeriodsPerThread = args.getSwitchArgument< unsigned int >("-pt");
    nrBinsPerThread = args.getSwitchArgument< unsigned int >("-bt");
    observation.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dms"));
    observation.setNrPeriods(args.getSwitchArgument< unsigned int >("-periods"));
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
    observation.setFirstPeriod(args.getSwitchArgument< unsigned int >("-first_period"));
    observation.setPeriodStep(args.getSwitchArgument< unsigned int >("-period_step"));
	} catch  ( isa::Exceptions::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-avx] [-phi] ... -padding ... -dt ... -pt ... -bt ... -seconds .... -samples ... -dms ... -periods ... -bins ... -first_period ... -period_step ..." << std::endl;
		return 1;
	}

  std::vector< unsigned int > * samplesPerBin = PulsarSearch::getSamplesPerBin(observation);

	// Allocate memory
  std::vector< std::vector< float > * > dedispersedData = std::vector< std::vector< float > * >(observation.getNrSeconds());
  std::vector< std::vector< float > * > dedispersedData_c = std::vector< std::vector< float > * >(observation.getNrSeconds());
  for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
    dedispersedData.at(second) = new std::vector< float >(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
    dedispersedData_c.at(second) = new std::vector< float >(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond());
  }
  std::vector< float > foldedData_c = std::vector< float >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  std::vector< float > foldedData = std::vector< float >(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
  std::vector< unsigned int > counters_c = std::vector< unsigned int >(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
  std::vector< unsigned int > readCounters = std::vector< unsigned int >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  std::vector< unsigned int > writeCounters = std::vector< unsigned int >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());

	srand(time(NULL));
  for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
        dedispersedData.at(second)[(sample * observation.getNrPaddedDMs()) + dm] = static_cast< float >(rand() % 10);
        dedispersedData_c.at(second)[(dm * observation.getNrSamplesPerPaddedSecond()) + sample] = dedispersedData.at(second)[(sample * observation.getNrPaddedDMs()) + dm];
      }
    }
  }
  std::fill(foldedData.begin(), foldedData.end(), static_cast< float >(0));
  std::fill(foldedData_c.begin(), foldedData_c.end(), static_cast< float >(0));
  std::fill(counters_c.begin(), counters_c.end(), 0);
  std::fill(readCounters.begin(), readCounters.end(), 0);
  std::fill(writeCounters.begin(), writeCounters.end(), 0);

  // Generate kernel
  PulsarSearch::foldingFunc< float > folding = 0;

  if ( avx ) {
    folding = functionPointers->at("foldingAVX" + isa::utils::toString< unsigned int >(nrDMsPerThread) + "x" + isa::utils::toString< unsigned int >(nrPeriodsPerThread) + "x" + isa::utils::toString< unsigned int >(nrBinsPerThread));
  } else if ( phi ) {
    folding = functionPointers->at("foldingPhi" + isa::utils::toString< unsigned int >(nrDMsPerThread) + "x" + isa::utils::toString< unsigned int >(nrPeriodsPerThread) + "x" + isa::utils::toString< unsigned int >(nrBinsPerThread));
  }

  // Run SIMD kernel and CPU control
  for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
    folding(second, observation, dedispersedData.at(second)->data(), foldedData.data(), readCounters.data(), writeCounters.data(), samplesPerBin->data());
    PulsarSearch::folding(second, observation, *(dedispersedData_c.at(second)), foldedData_c, counters_c);
  }

  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
      for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
        if ( ! same(foldedData_c[(dm * observation.getNrPeriods() * observation.getNrPaddedBins()) + (period * observation.getNrPaddedBins()) + bin], foldedData[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) * (period * observation.getNrPaddedDMs()) + dm]) ) {
          wrongSamples++;
        }
      }
    }
  }

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins()) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

