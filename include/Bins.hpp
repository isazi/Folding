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

#include <vector>
using std::vector;

#include <Observation.hpp>
using AstroData::Observation;


#ifndef BINS_HPP
#define BINS_HPP

namespace PulsarSearch {

template< typename T > vector< unsigned int > * getNrSamplesPerBin(const Observation< T > & obs);

// Implementation
template< typename T > vector< unsigned int > * getNrSamplesPerBin(const Observation< T > & obs) {
	vector< unsigned int > * samplesPerBin = new vector< unsigned int >(obs.getNrPeriods() * 2 * obs.getNrPaddedBins());

	for ( unsigned int period = 0; period < obs.getNrPeriods(); period++ ) {
		unsigned int offset = 0;
		unsigned int periodValue = obs.getFirstPeriod() + (period * obs.getPeriodStep());

		for ( unsigned int bin = 0; bin < obs.getNrBins(); bin++ ) {
			samplesPerBin->at((period * 2 * obs.getNrPaddedBins()) + (bin * 2)) = periodValue / obs.getNrBins();
			if ( ((periodValue % obs.getNrBins()) != 0) && ((bin % (obs.getNrBins() / (periodValue % obs.getNrBins()))) == 0) ) {
				samplesPerBin->at((period * 2 * obs.getNrPaddedBins()) + (bin * 2)) += 1;
			}
			samplesPerBin->at((period * 2 * obs.getNrPaddedBins()) + (bin * 2) + 1) = offset;
			offset += samplesPerBin->at((period * 2 * obs.getNrPaddedBins()) + (bin * 2));
		}
	}

	return samplesPerBin;
}

} // PulsarSearch

#endif // BINS_HPP
