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
#include <algorithm>

#include <Observation.hpp>
#include <utils.hpp>


#ifndef BINS_HPP
#define BINS_HPP

namespace PulsarSearch {

template< typename T > std::vector< unsigned int > * getNrSamplesPerBin(const AstroData::Observation< T > & obs);

// Implementation
template< typename T > std::vector< unsigned int > * getNrSamplesPerBin(const AstroData::Observation< T > & obs) {
  std::vector< unsigned int > * samplesPerBin = new std::vector< unsigned int >(obs.getNrPeriods() * obs.getNrBins() * isa::utils::pad(2, obs.getPadding()));

	for ( unsigned int period = 0; period < obs.getNrPeriods(); period++ ) {
		unsigned int offset = 0;
    std::vector< unsigned int > itemsPerBin;
    std::vector< unsigned int > offsetPerBin;

    std::fill(itemsPerBin.begin(), itemsPerBin.end(), 0);
    std::fill(offsetPerBin.begin(), offsetPerBin.end(), 0);
    for ( unsigned int i = 0; i < period; i++ ) {
      float samplePhase = (i / period);

      for ( unsigned int bin = 0; bin < obs.getNrBins(); bin++ ) {
        if ( samplePhase - ((bin + 0.5) / obs.getNrBins()) < 1.0f / obs.getNrBins() ) {
          itemsPerBin[bin] += 1;
          break;
        }
      }
    }

		for ( unsigned int bin = 0; bin < obs.getNrBins(); bin++ ) {
      samplesPerBin->at((period * obs.getNrBins() * pad(2, obs.getPadding())) + (bin * pad(2, obs.getPadding()))) = itemsPerBin[bin];
      samplesPerBin->at((period * obs.getNrBins() * pad(2, obs.getPadding())) + (bin * pad(2, obs.getPadding())) + 1) = offset;
      offset += itemsPerBin[bin];
    }
	}

	return samplesPerBin;
}

} // PulsarSearch

#endif // BINS_HPP
