/*
 * Copyright (C) 2013
 * Alessio Sclocco <a.sclocco@vu.nl>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <string>
#include <vector>
#include <cmath>
#include <x86intrin.h>
using std::string;
using std::vector;
using std::make_pair;
using std::pow;
using std::ceil;

#include <Observation.hpp>
using AstroData::Observation;


#ifndef FOLDING_PHI_HPP
#define FOLDING_PHI_HPP

namespace PulsarSearch {

// OpenMP folding algorithm
template< typename T > void folding(const unsigned int second, const unsigned int nrDMs, const unsigned int nrPeriods, const unsigned int firstPeriod, const unsigned int periodStep, const unsigned int nrSamplesPerSecond, const unsigned int nrSamplesPerPaddedSecond, const unsigned int nrBins, const unsigned int nrPaddedBins, const T * const __restrict__ samples, T * const __restrict__ bins, unsigned int * const __restrict__ counters);


// Implementation
template< typename T > void folding(const unsigned int second, const unsigned int nrDMs, const unsigned int nrPeriods, const unsigned int firstPeriod, const unsigned int periodStep, const unsigned int nrSamplesPerSecond, const unsigned int nrSamplesPerPaddedSecond, const unsigned int nrBins, const unsigned int nrPaddedBins, const T * const __restrict__ samples, T * const __restrict__ bins, unsigned int * const __restrict__ counters) {
	#pragma offload target(mic) nocopy(samples: alloc_if(0) free_if(0)) nocopy(bins: alloc_if(0) free_if(0)) nocopy(counters: alloc_if(0) free_if(0))
	{
		#pragma omp parallel for
		for ( unsigned int dm = 0; dm < nrDMs; dm++ ) {
			#pragma omp parallel for
			for ( unsigned int periodIndex = 0; periodIndex < nrPeriods; periodIndex++ ) {
				const unsigned int periodValue = firstPeriod + ( periodIndex * periodStep );
				
				for ( unsigned int globalSample = 0; globalSample < nrSamplesPerSecond; globalSample++ ) {
					const unsigned int sample = ( second * nrSamplesPerSecond ) + globalSample;
					const float phase = ( sample / static_cast< float >(periodValue) ) - ( sample / periodValue );
					const unsigned int bin = static_cast< unsigned int >(phase * static_cast< float >(nrBins));
					const unsigned int globalItem = ( ( ( dm * nrPeriods ) + periodIndex ) * nrPaddedBins ) + bin;
	
					const T pValue = bins[globalItem];
					T cValue = samples[( dm * nrSamplesPerPaddedSecond ) + globalSample];
					const unsigned int pCounter = counters[globalItem];
					unsigned int cCounter = pCounter + 1;
	
					if ( pCounter != 0 ) {
						cValue = pValue + ( ( cValue - pValue ) / cCounter );
					}
					bins[globalItem] = cValue;
					counters[globalItem] = cCounter;
				}
			}
		}
	}
}

} // PulsarSearch

#endif // FOLDING_PHI_HPP