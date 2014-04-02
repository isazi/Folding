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

#include <string>
using std::string;
#include <vector>
using std::vector;
#include <cmath>
using std::pow;
using std::ceil;
#include <x86intrin.h>
#include <omp.h>

#include <Observation.hpp>
using AstroData::Observation;
#include <Bins.hpp>
using PulsarSearch::getNrSamplesPerBin;


#ifndef FOLDING_PHI_HPP
#define FOLDING_PHI_HPP

namespace PulsarSearch {

// OpenMP folding algorithm
template< typename T > void folding(const unsigned int second, const Observation< T > & observation, const T * const __restrict__ samples, T * const __restrict__ bins, unsigned int * const __restrict__ writeCounters, const unsigned int * const __restrict__ readCounters);


// Implementation
template< typename T > void folding(const unsigned int second, const Observation< T > & observation, const T * const __restrict__ samples, T * const __restrict__ bins, unsigned int * const __restrict__ writeCounters, const unsigned int * const __restrict__ readCounters) {
	vector< unsigned int > * samplesPerBin = getNrSamplesPerBin(observation);

	#pragma omp parallel for schedule(static)
	for ( unsigned int periodIndex = 0; periodIndex < observation.getNrPeriods(); periodIndex++ ) {
		const unsigned int periodValue = observation.getFirstPeriod() + (periodIndex * observation.getPeriodStep());

		#pragma omp parallel for schedule(static)
		for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
			const unsigned int pCounter = readCounters[(periodIndex * observation.getNrPaddedBins()) + bin];

			for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm += 8 ) {
				__m512 foldedSample = _mm512_setzero_ps();
				unsigned int foldedCounter = 0;
				unsigned int sample = samplesPerBin->at((periodIndex * 2 * observation.getNrPaddedBins()) + (bin * 2) + 1) + ((pCounter / samplesPerBin->at((periodIndex * 2 * observation.getNrPaddedBins()) + (bin * 2))) * periodValue) + (pCounter % samplesPerBin->at((periodIndex * 2 * observation.getNrPaddedBins()) + (bin * 2)));

				if ( (sample / observation.getNrSamplesPerSecond()) == second ) {
					sample %= observation.getNrSamplesPerSecond();
				}
				while ( sample < observation.getNrSamplesPerSecond() ) {
					foldedSample = _mm512_add_ps(foldedSample, _mm512_load_ps(&(samples[(sample * observation.getNrPaddedDMs()) + dm])));
					foldedCounter++;

					if ( (foldedCounter + pCounter) % samplesPerBin->at((periodIndex * 2 * observation.getNrPaddedBins()) + (bin * 2)) == 0 ) {
						sample += periodValue;
					} else {
						sample++;
					}
				}

				if ( foldedCounter > 0 ) {
					const __m512 pValue = _mm512_load_ps(&(bins[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (periodIndex * observation.getNrPaddedDMs()) + dm]));
					float addedFraction = static_cast< float >(foldedCounter) / (foldedCounter + pCounter);

					foldedSample = _mm512_div_ps(foldedSample, _mm512_set1_ps(static_cast< float >(foldedCounter)));
					foldedSample = _mm512_mul_ps(foldedSample, _mm512_set1_ps(addedFraction));
					_mm512_store_ps(&(bins[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (periodIndex * observation.getNrPaddedDMs()) + dm]), _mm512_add_ps(foldedSample, _mm512_mul_ps(pValue, _mm512_set1_ps(1.0f - addedFraction))));
					writeCounters[(periodIndex * observation.getNrPaddedBins()) + bin] = pCounter + foldedCounter;
				}
			}
		}
	}
}

} // PulsarSearch

#endif // FOLDING_PHI_HPP
