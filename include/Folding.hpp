/*
 * Copyright (C) 2012
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

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
using std::string;
using std::vector;
using std::make_pair;
using std::pow;
using std::ceil;

#include <Exceptions.hpp>
#include <CLData.hpp>
#include <utils.hpp>
#include <Kernel.hpp>
#include <Observation.hpp>
using isa::Exceptions::OpenCLError;
using isa::OpenCL::CLData;
using isa::utils::giga;
using isa::utils::toStringValue;
using isa::utils::toString;
using isa::utils::replace;
using isa::OpenCL::Kernel;
using AstroData::Observation;


#ifndef FOLDING_HPP
#define FOLDING_HPP

namespace PulsarSearch {

// OpenCL folding algorithm
template< typename T > class Folding : public Kernel< T > {
public:
	Folding(string name, string dataType);
	
	void generateCode() throw (OpenCLError);
	void operator()(CLData< T > * input, CLData< T > * output, CLData< unsigned int > * counters) throw (OpenCLError);

	inline void setNrDMsPerBlock(unsigned int DMs);
	inline void setNrPeriodsPerBlock(unsigned int periods);
	inline void setNrBinsPerBlock(unsigned int bins);

	inline void setObservation(Observation< T > * obs);

private:
	unsigned int nrDMsPerBlock;
	unsigned int nrPeriodsPerBlock;
	unsigned int nrBinsPerBlock;
	cl::NDRange globalSize;
	cl::NDRange localSize;

	Observation< T > * observation;
};


// Implementation
template< typename T > Folding< T >::Folding(string name, string dataType) : Kernel< T >(name, dataType), nrDMsPerBlock(0), nrPeriodsPerBlock(0), nrBinsPerBlock(0), globalSize(cl::NDRange(1, 1, 1)), localSize(cl::NDRange(1, 1, 1)), observation(0) {}

template< typename T > void Folding< T >::generateCode() throw (OpenCLError) {
	// Begin kernel's template
	string nrSamplesPerSecond_s = toStringValue< unsigned int >(observation->getNrSamplesPerSecond());
	string nrPaddedDMs_s  = toStringValue< unsigned int >(observation->getNrPaddedDMs());
	string nrPeriods_s = toStringValue< unsigned int >(observation->getNrPeriods());
	string nrDMsPerBlock_s = toStringValue< unsigned int >(nrDMsPerBlock);
	string nrPeriodsPerBlock_s = toStringValue< unsigned int >(nrPeriodsPerBlock);
	string nrBinsPerBlock_s = toStringValue< unsigned int >(nrBinsPerBlock);
	string nrBins_s = toStringValue< unsigned int >(observation->getNrBins());

	delete this->code;
	this->code = new string();
	*(this->code) = "__kernel void " + this->name + "(__global const " + this->dataType + " * const restrict samples, __global " + this->dataType + " * const restrict bins, __global unsigned int * const restrict counters) {\n"
	"const unsigned int DM = (get_group_id(0) * " + nrDMsPerBlock_s + ") + get_local_id(0);\n"
	"const unsigned int period = (get_group_id(1) * " + nrPeriodsPerBlock_s + ") + get_local_id(1);\n"
	"const unsigned int bin = (get_group_id(2) * " + nrBinsPerBlock_s + ") + get_local_id(2);\n"
	"const unsigned int periodValue = (period + 1) * " + nrBins_s + ";\n"
	"unsigned int foldedCounter = 0;\n"
	+ this->dataType + " foldedSample = 0;\n"
	"const unsigned int pCounter = counters[(bin * " + nrPeriods_s + " * " + nrPaddedDMs_s + ") + (period * " + nrPaddedDMs_s + ") + DM];\n"

	"\n"
	"unsigned int sample = (bin * period) + bin + ((pCounter / (period + 1)) * periodValue) + (pCounter % (period + 1));\n"
	"if ( (sample % "+ nrSamplesPerSecond_s + ") == 0 ) {\n"
	"sample = 0;\n"
	"} else {\n"
	"sample = (sample % "+ nrSamplesPerSecond_s + ") - (sample / "+ nrSamplesPerSecond_s + ");\n"
	"}\n"
	"while ( sample < " + nrSamplesPerSecond_s + " ) {\n"
	"foldedSample += samples[(sample * " + nrPaddedDMs_s + ") + DM];\n"
	"foldedCounter++;\n"
	"if ( ((foldedCounter + pCounter) % (period + 1)) == 0 ) {\n"
	"sample += periodValue;\n"
	"} else {\n"
	"sample++;\n"
	"}\n"
	"}\n"
	"if ( foldedCounter > 0 ) {\n"
	"const unsigned int outputItem = (bin * " + nrPeriods_s + " * " + nrPaddedDMs_s + ") + (period * " + nrPaddedDMs_s + ") + DM;\n"
	"const "+ this->dataType + " pValue = bins[outputItem];\n"
	+ this->dataType + " addedFraction = 0;"
	"addedFraction = convert_float(foldedCounter) / (foldedCounter + pCounter);\n"
	"foldedSample /= foldedCounter;\n"
	"counters[outputItem] = pCounter + foldedCounter;\n"
	"bins[outputItem] = (addedFraction * foldedSample) + ((1.0f - addedFraction) * pValue);\n"
	"}\n"
	"}\n";

	globalSize = cl::NDRange(observation->getNrDMs(), observation->getNrPeriods(), observation->getNrBins());
	localSize = cl::NDRange(nrDMsPerBlock, nrPeriodsPerBlock, nrBinsPerBlock);

	this->gflop = giga(static_cast< long long unsigned int >(observation->getNrDMs()) * observation->getNrPeriods() * observation->getNrSamplesPerSecond());

	this->compile();
}

template< typename T > void Folding< T >::operator()(CLData< T > * input, CLData< T > * output, CLData< unsigned int > * counters) throw (OpenCLError) {
	this->setArgument(0, *(input->getDeviceData()));
	this->setArgument(1, *(output->getDeviceData()));
	this->setArgument(2, *(counters->getDeviceData()));

	this->run(globalSize, localSize);
}

template< typename T > inline void Folding< T >::setNrDMsPerBlock(unsigned int DMs) {
	nrDMsPerBlock = DMs;
}

template< typename T > inline void Folding< T >::setNrPeriodsPerBlock(unsigned int periods) {
	nrPeriodsPerBlock = periods;
}

template< typename T > inline void Folding< T >::setNrBinsPerBlock(unsigned int bins) {
	nrBinsPerBlock = bins;
}

template< typename T > inline void Folding< T >::setObservation(Observation< T > * obs) {
	observation = obs;
}

} // PulsarSearch

#endif // FOLDING_HPP
