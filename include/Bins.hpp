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

#include <vector>
#include <algorithm>

#include <Observation.hpp>
#include <utils.hpp>


#ifndef BINS_HPP
#define BINS_HPP

namespace PulsarSearch {

template< typename T > std::vector< unsigned int > * getSamplesPerBin(const AstroData::Observation< T > & obs);

// Implementation
template< typename T > std::vector< unsigned int > * getSamplesPerBin(const AstroData::Observation< T > & obs) {
  std::vector< unsigned int > * samplesPerBin = new std::vector< unsigned int >(obs.getNrPeriods() * obs.getNrBins() * isa::utils::pad(2, obs.getPadding()));

  for ( unsigned int period = 0; period < obs.getNrPeriods(); period++ ) {
		unsigned int offset = 0;
    unsigned int periodValue = (obs.getFirstPeriod() + (obs.getPeriodStep() * period));
    std::vector< unsigned int > itemsPerBin(obs.getNrBins());
    std::vector< unsigned int > offsetPerBin(obs.getNrBins());

    std::fill(itemsPerBin.begin(), itemsPerBin.end(), 0);
    std::fill(offsetPerBin.begin(), offsetPerBin.end(), 0);
    for ( unsigned int sample = 0; sample < periodValue; sample++ ) {
      float samplePhase = (static_cast< float >(sample) / periodValue);
      const unsigned int bin = static_cast< unsigned int >(samplePhase * obs.getNrBins());
      
      itemsPerBin[bin] += 1;
    }

		for ( unsigned int bin = 0; bin < obs.getNrBins(); bin++ ) {
      samplesPerBin->at((period * obs.getNrBins() * isa::utils::pad(2, obs.getPadding())) + (bin * isa::utils::pad(2, obs.getPadding()))) = itemsPerBin[bin];
      samplesPerBin->at((period * obs.getNrBins() * isa::utils::pad(2, obs.getPadding())) + (bin * isa::utils::pad(2, obs.getPadding())) + 1) = offset;
      offset += itemsPerBin[bin];
    }
	}

	return samplesPerBin;
}

} // PulsarSearch

#endif // BINS_HPP
