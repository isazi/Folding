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

#include <Observation.hpp>
using AstroData::Observation;


#ifndef BINS_HPP
#define BINS_HPP

namespace PulsarSearch {

template< typename T > unsigned int * getNrSamplesPerBin(Observation< T > & obs);

// Implementation
template< typename T > unsigned int * getNrSamplesPerBin(Observation< T > & obs) {
	unsigned int * samplesPerBin = new unsigned int [obs.getNrPeriods() * obs.getNrPaddedBins()];

	for ( unsigned int period = 0; period < obs.getNrPeriods(); period++ ) {
		unsigned int periodValue = obs.getFirstPeriod() + (period * obs.getPeriodStep());

		for ( unsigned int bin = 0; bin < obs.getNrBins(); bin++ ) {
			samplesPerBin[(period * obs.getNrPaddedBins()) + bin] = periodValue / obs.getBasePeriod();
		}
		for ( unsigned int bin = 0; bin < periodValue % obs.getBasePeriod(); bin++ ) {
			samplesPerBin[(period * obs.getNrPaddedBins()) + bin] += 1;
		}
	}

	return samplesPerBin;
}

} // PulsarSearch

#endif // BINS_HPP
