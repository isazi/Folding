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


#ifndef FOLDING_AVX_HPP
#define FOLDING_AVX_HPP

namespace PulsarSearch {

// OpenMP folding algorithm
void folding(const unsigned int second, const Observation< float > & observation, const float * const __restrict__ samples, float * const __restrict__ bins, const unsigned int * const __restrict__ readCounters, unsigned int * const __restrict__ writeCounters);


// Implementation
void folding(const unsigned int second, const Observation< float > & observation, const float * const __restrict__ samples, float * const __restrict__ bins, const unsigned int * const __restrict__ readCounters, unsigned int * const __restrict__ writeCounters) {
	vector< unsigned int > * samplesPerBin = getNrSamplesPerBin(observation);

	#pragma omp parallel for schedule(static)
	for ( unsigned int periodIndex = 0; periodIndex < observation.getNrPeriods(); periodIndex++ ) {
		const unsigned int periodValue = observation.getFirstPeriod() + (periodIndex * observation.getPeriodStep());

		#pragma omp parallel for schedule(static)
		for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
			const unsigned int pCounter = readCounters[(periodIndex * observation.getNrPaddedBins()) + bin];
			
			for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm += 8 ) {
				__m256 foldedSample = _mm256_setzero_ps();
				unsigned int foldedCounter = 0;
				unsigned int sample = samplesPerBin->at((periodIndex * 2 * observation.getNrPaddedBins()) + (bin * 2) + 1) + ((pCounter / samplesPerBin->at((periodIndex * 2 * observation.getNrPaddedBins()) + (bin * 2))) * periodValue) + (pCounter % samplesPerBin->at((periodIndex * 2 * observation.getNrPaddedBins()) + (bin * 2)));

				if ( (sample / observation.getNrSamplesPerSecond()) == second ) {
					sample %= observation.getNrSamplesPerSecond();
				}	
				while ( sample < observation.getNrSamplesPerSecond() ) {
					foldedSample = _mm256_add_ps(foldedSample, _mm256_load_ps(&(samples[(sample * observation.getNrPaddedDMs()) + dm])));
					foldedCounter++;

					if ( (foldedCounter + pCounter) % samplesPerBin->at((periodIndex * 2 * observation.getNrPaddedBins()) + (bin * 2)) == 0 ) {
						sample += periodValue;
					} else {
						sample++;
					}
				}

				if ( foldedCounter > 0 ) {
					const __m256 pValue = _mm256_load_ps(&(bins[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (periodIndex * observation.getNrPaddedDMs()) + dm]));
					float addedFraction = static_cast< float >(foldedCounter) / (foldedCounter + pCounter);

					foldedSample = _mm256_div_ps(foldedSample, _mm256_set1_ps(static_cast< float >(foldedCounter)));
					foldedSample = _mm256_mul_ps(foldedSample, _mm256_set1_ps(addedFraction));
					_mm256_store_ps(&(bins[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (periodIndex * observation.getNrPaddedDMs()) + dm]), _mm256_add_ps(foldedSample, _mm256_mul_ps(pValue, _mm256_set1_ps(1.0f - addedFraction))));
					writeCounters[(periodIndex * observation.getNrPaddedBins()) + bin] = pCounter + foldedCounter;
				}
			}
		}
	}
}

} // PulsarSearch

#endif // FOLDING_AVX_HPP