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
#include <Observation.hpp>
#include <utils.hpp>
#include <Bins.hpp>

typedef float dataType;
std::string typeName("float");


int main(int argc, char *argv[]) {
  bool print = false;
	long long unsigned int wrongSamples = 0;
	AstroData::Observation observation;

	try {
    isa::utils::ArgumentList args(argc, argv);
    observation.setPadding(1);
    observation.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
    observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
    observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), args.getSwitchArgument< unsigned int >("-first_period"), args.getSwitchArgument< unsigned int >("-period_step"));
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
	} catch  ( isa::utils::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " -samples ... -dms ... -periods ... -bins ... -first_period ... -period_step ..." << std::endl;
		return 1;
	}

  std::vector< unsigned int > * samplesPerBin = PulsarSearch::getSamplesPerBin(observation);

  // Allocate memory
  std::vector< unsigned int > sequentialMap(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrSeconds() * observation.getNrSamplesPerSecond());
  std::vector< unsigned int > parallelMap(sequentialMap.size());
  std::vector< unsigned int > parallelCounter(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrBins());

  std::fill(parallelCounter.begin(), parallelCounter.end(), 0);

  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
      const unsigned int periodValue = observation.getFirstPeriod() + (period * observation.getPeriodStep());

      for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
        for ( unsigned int sample = 0; second < observation.getNrSamplesPerSecond(); sample++ ) {
          const unsigned int globalSample = (second * observation.getNrSamplesPerSecond()) + sample;
          const float phase = (globalSample / static_cast< float >(periodValue)) - (globalSample / periodValue);
          const unsigned int bin = static_cast< unsigned int >(phase * static_cast< float >(observation.getNrBins()));

          sequentialMap[(dm * observation.getNrPeriods() * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + (period * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + globalSample] = bin;
        }
      }
    }
  }

  for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
      const unsigned int periodValue = observation.getFirstPeriod() + (period * observation.getPeriodStep());

      for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
          unsigned int sample = samplesPerBin->at((period * 2 * observation.getNrPaddedBins()) + (bin * 2) + 1) + ((parallelCounter[(dm * observation.getNrPeriods() * observation.getNrBins()) + (period * observation.getNrBins()) + bin] / samplesPerBin->at((period * 2 * observation.getNrPaddedBins()) + (bin * 2))) * periodValue) + (parallelCounter[(dm * observation.getNrPeriods() * observation.getNrBins()) + (period * observation.getNrBins()) + bin] % samplesPerBin->at((period * 2 * observation.getNrPaddedBins()) + (bin * 2)));

          if ( (sample / observation.getNrSamplesPerSecond()) == second ) {
            sample %= observation.getNrSamplesPerSecond();
          }
          while ( sample < observation.getNrSamplesPerSecond() ) {
            parallelMap[(dm * observation.getNrPeriods() * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + (period * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + sample] = bin;
            parallelCounter[(dm * observation.getNrPeriods() * observation.getNrBins()) + (period * observation.getNrBins()) + bin] += 1;
            if ( parallelCounter[(dm * observation.getNrPeriods() * observation.getNrBins()) + (period * observation.getNrBins()) + bin] % samplesPerBin->at((period * 2 * observation.getNrPaddedBins()) + (bin * 2)) == 0 ) {
              sample += periodValue;
            } else {
              sample++;
            }
          }
        }
      }
    }
  }

  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
      for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
        for ( unsigned int sample = 0; second < observation.getNrSamplesPerSecond(); sample++ ) {
          if ( sequentialMap[(dm * observation.getNrPeriods() * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + (period * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + sample] != parallelMap[(dm * observation.getNrPeriods() * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + (period * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + sample] ) {
            std::cout << "DM: " << dm << ", ";
            std::cout << "Period: " << period << ", ";
            std::cout << "Second: " << second << ", ";
            std::cout << "Sample: " << sample << ", ";
            std::cout << "Bin (seq): " << sequentialMap[(dm * observation.getNrPeriods() * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + (period * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + sample] << ", ";
            std::cout << "Bin (par): " << parallelMap[(dm * observation.getNrPeriods() * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + (period * observation.getNrSeconds() * observation.getNrSamplesPerSecond()) + sample] << std::endl;
          }
        }
      }
    }
  }
	return 0;
}

