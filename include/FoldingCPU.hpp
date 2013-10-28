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
#include <omp.h>
using std::string;
using std::vector;
using std::make_pair;
using std::pow;
using std::ceil;

#include <Observation.hpp>
using AstroData::Observation;


#ifndef FOLDING_CPU_HPP
#define FOLDING_CPU_HPP

namespace TDM {

// OpenMP folding algorithm
template< typename T > void folding(const unsigned int second, const Observation< T > & observation, const T * const __restrict__ samples, T * const __restrict__ bins, unsigned int * const __restrict__ counters);


// Implementation
template< typename T > void folding(const unsigned int second, const Observation< T > & observation, const T * const __restrict__ samples, T * const __restrict__ bins, unsigned int * const __restrict__ counters) {
	#pragma omp parallel for
	for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
		#pragma omp parallel for
		for ( unsigned int periodIndex = 0; periodIndex < observation.getNrPeriods(); periodIndex++ ) {
			const unsigned int periodValue = observation.getFirstPeriod() + ( periodIndex * observation.getPeriodStep() );
			
			for ( unsigned int globalSample = 0; globalSample < observation.getNrSamplesPerSecond(); globalSample++ ) {
				const unsigned int sample = ( second * observation.getNrSamplesPerSecond() ) + globalSample;
				const float phase = ( sample / static_cast< float >(periodValue) ) - ( sample / periodValue );
				const unsigned int bin = static_cast< unsigned int >(phase * static_cast< float >(observation.getNrBins()));
				const unsigned int globalItem = ( ( ( dm * observation.getNrPeriods() ) + periodIndex ) * observation.getNrPaddedBins() ) + bin;

				const T pValue = bins[globalItem];
				T cValue = samples[( dm * observation.getNrSamplesPerPaddedSecond() ) + globalSample];
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

} // TDM

#endif // FOLDING_CPU_HPP