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

		for ( unsigned int bin = 0; bin < obs.getNrBins(); bin += 2 ) {
			samplesPerBin->at((period * 2 * obs.getNrPaddedBins()) + bin) = periodValue / obs.getNrBins();
			if ( bin < periodValue % obs.getNrBins() ) {
				samplesPerBin->at((period * 2 * obs.getNrPaddedBins()) + bin) += 1;
			}
			samplesPerBin->at((period * 2 * obs.getNrPaddedBins()) + bin + 1) = offset;
			offset += samplesPerBin->at((period * 2 * obs.getNrPaddedBins()) + bin);
		}
	}

	return samplesPerBin;
}

} // PulsarSearch

#endif // BINS_HPP
