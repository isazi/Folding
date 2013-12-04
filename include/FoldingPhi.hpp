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