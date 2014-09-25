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
#include <algorithm>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <Bins.hpp>
#include <Folding.hpp>
#include <utils.hpp>
#include <Exceptions.hpp>
#include <Timer.hpp>
#include <Stats.hpp>

typedef float dataType;
string typeName("float");


int main(int argc, char * argv[]) {
  bool avx = false;
  bool phi = false;
	unsigned int nrIterations = 0;
	unsigned int maxItemsPerThread = 0;
  AstroData::Observation observation;

	try {
    isa::utils::ArgumentList args(argc, argv);

    avx = args.getSwitch("=avx");
    phi = args.getSwitch("-phi");
    if ( !(avx || phi) ) {
      throw isa::Exceptions::EmptyCommandLine();
    }
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
		maxItemsPerThread = args.getSwitchArgument< unsigned int >("-max_items");
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
    observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
    observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), args.getSwitchArgument< unsigned int >("-first_period"), args.getSwitchArgument< unsigned int >("-period_step"));
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
	} catch ( isa::Exceptions::EmptyCommandLine &err ) {
		std::cerr << argv[0] << " [-avx] [-phi] -iterations ... -padding ... -max_items ... -samples ... -dms ... -periods ... -bins ... -first_period ... -period_step ..." << std::endl;
		return 1;
	} catch ( std::exception &err ) {
		std::cerr << err.what() << std::endl;
		return 1;
	}

  std::vector< unsigned int > * samplesPerBin = PulsarSearch::getSamplesPerBin(observation);

	// Allocate memory
  std::vector< float > dedispersedData = std::vector< float >(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
  std::vector< float > foldedData = std::vector< float >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  std::vector< unsigned int > readCounters = std::vector< unsigned int >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  std::vector< unsigned int > writeCounters = std::vector< unsigned int >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());

	srand(time(NULL));
  for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
    for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
      dedispersedData[(sample * observation.getNrPaddedDMs()) + DM] = static_cast< float >(rand() % 10);
		}
	}
  std::fill(foldedData.begin(), foldedData.end(), static_cast< float >(0));
  std::fill(readCounters.begin(), readCounters.end(), 0);
  std::fill(writeCounters.begin(), writeCounters.end(), 0);

	std::cout << std::fixed << std::endl;
	std::cout << "# nrDMs nrSamples nrPeriods nrBins firstPeriod periodStep DMsPerThread periodsPerThread binsPerThread GFLOP/s err time err" << std::endl << std::endl;

  for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
    if ( avx ){
      if ( observation.getNrPaddedDMs() % (DMsPerThread * 8) != 0 ) {
        continue;
      }
    } else ( phi ) {
      if ( observation.getNrPaddedDMs() % (DMsPerThread * 16) != 0 ) {
        continue;
      }
    }
    for ( unsigned int periodsPerThread = 1; periodsPerThread <= maxItemsPerThread; periodsPerThread++ ) {
      if ( observation.getNrPeriods() % periodsPerThread != 0 ) {
        continue;
      }
      for ( unsigned int binsPerThread = 1; binsPerThread <= maxItemsPerThread; binsPerThread++ ) {
        if ( observation.getNrBins() % binsPerThread != 0 ) {
          continue;
        }
        if ( periodsPerThread + (3 * periodsPerThread * binsPerThread) + (periodsPerThread * binsPerThread * DMsPerThread) > maxItemsPerThread ) {
          break;
        }

        // Tuning runs
        double flops = isa::utils::giga(static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrSamplesPerSecond());
        isa::utils::Timer timer;
        isa::utils::Stats< double > stats;
        PulsarSearch::foldingFunc< float > folding = 0;

        if ( avx ) {
          folding = functionPointers->at("foldingAVX" + isa::utils::toString< unsigned int >(DMsPerThread) + "x" + isa::utils::toString< unsigned int >(periodsPerThread) + "x" + isa::utils::toString< unsigned int >(binsPerThread));
        } else if ( phi ) {
          folding = functionPointers->at("foldingPhi" + isa::utils::toString< unsigned int >(DMsPerThread) + "x" + isa::utils::toString< unsigned int >(periodsPerThread) + "x" + isa::utils::toString< unsigned int >(binsPerThread));
        }
        for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
          std::memcpy(dedispersedData.data(), dedispersedData.data(), dedispersedData.size() * sizeof(float));
          std::memcpy(readCounters.data(), readCounters.data(), readCounters.size() * sizeof(unsigned int));
          std::memcpy(writeCounters.data(), writeCounters.data(), writeCounters.size() * sizeof(unsigned int));
          timer.start();
          folding(0, observation, dedispersedData.data(), foldedData.data(), readCounters.data(), writeCounters.data(), samplesPerBin->data());
          timer.stop();
          stats.addElement(flops / timer.getLastRunTime());
        }

        std::cout << observation.getNrDMs() << " " << observation.getNrSamplesPerSecond() << " " << observation.getNrPeriods() << " " << observation.getNrBins() << " " << observation.getFirstPeriod() << " " << observation.getPeriodStep() << " " << DMsPerThread << " " << periodsPerThread << " " << binsPerThread << " " << std::setprecision(3) << stats.getAverage() << " " << stats.getStdDev() << " " << std::setprecision(6) << timer.getAverageTime() << " " << timer.getStdDev() << std::endl;
      }
    }
  }

	std::cout << std::endl;

	return 0;
}

