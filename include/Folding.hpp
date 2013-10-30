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
	void operator()(unsigned int second, CLData< T > * input, CLData< T > * output, CLData< unsigned int > * counters) throw (OpenCLError);

	inline void setNrPeriodsPerBlock(unsigned int periods);

	inline void setObservation(Observation< T > * obs);

private:
	unsigned int nrPeriodsPerBlock;
	cl::NDRange globalSize;
	cl::NDRange localSize;

	Observation< T > * observation;
};


// Implementation
template< typename T > Folding< T >::Folding(string name, string dataType) : Kernel< T >(name, dataType), nrPeriodsPerBlock(0), globalSize(cl::NDRange(1, 1, 1)), localSize(cl::NDRange(1, 1, 1)), observation(0) {}

template< typename T > void Folding< T >::generateCode() throw (OpenCLError) {
	// Begin kernel's template
	string nrSamplesPerSecond_s = toStringValue< unsigned int >(observation->getNrSamplesPerSecond());
	string nrSamplesPerPaddedSecond_s = toStringValue< unsigned int >(observation->getNrSamplesPerPaddedSecond());
	string nrPaddedPeriods_s  = toStringValue< unsigned int >(observation->getNrPaddedPeriods());
	string nrPeriods_s = toStringValue< unsigned int >(observation->getNrPeriods());
	string firstPeriod_s = toStringValue< unsigned int >(observation->getFirstPeriod());
	string periodStep_s = toStringValue< unsigned int >(observation->getPeriodStep());
	string nrPeriodsPerBlock_s = toStringValue< unsigned int >(nrPeriodsPerBlock);
	string nrBins_s = toStringValue< unsigned int >(observation->getNrBins());

	delete this->code;
	this->code = new string();
	*(this->code) = "__kernel void " + this->name + "(const unsigned int second, __global const " + this->dataType + " * const restrict samples, __global " + this->dataType + " * const restrict bins, __global unsigned int * const restrict counters) {\n"
		"const unsigned int dm = get_group_id(1);\n"
		"const unsigned int periodIndex = (get_group_id(0) * " + nrPeriodsPerBlock_s + ") + get_local_id(0);\n"
		"const unsigned int period = " + firstPeriod_s + " + (periodIndex * " + periodStep_s + ");\n"
		"__local " + this->dataType + " block[" + nrPeriodsPerBlock_s + "];\n"
		"<%BINS%>"
		"<%COUNTERS%>"
		"\n"
		"for ( unsigned int globalSample = get_local_id(0); globalSample < " + nrSamplesPerPaddedSecond_s + "; globalSample += " + nrPeriodsPerBlock_s + " ) {\n"
		"block[get_local_id(0)] = samples[(dm * " + nrSamplesPerPaddedSecond_s + ") + globalSample];\n"
		"barrier(CLK_LOCAL_MEM_FENCE);\n"
		"for ( unsigned int localSample = 0; localSample < " + nrPeriodsPerBlock_s + "; localSample++ ) {\n"
		"unsigned int sample = ( globalSample - get_local_id(0) ) + localSample;\n"
		"const " + this->dataType + " cSample = block[localSample];\n"
		"if ( sample >= " + nrSamplesPerSecond_s + " ) {\n"
		"break;\n"
		"}\n"
		"sample += ( second * " + nrSamplesPerSecond_s + " );\n"
		"const float phase = ( sample / convert_float(period) ) - ( sample / period );\n"
		"const unsigned int bin = convert_uint_rtz(phase * " + nrBins_s + ".0f);\n"
		"\n"
		"<%ACCUMULATE%>"
		"}\n"
		"}\n"
		"\n"
		"unsigned int globalItem = 0;\n"
		"unsigned int pCounter = 0;\n"
		+ this->dataType + " pValue = convert_" + this->dataType + "(0);\n"
		"float addedFraction = 0.0f;\n"
		"<%STORE%>"
		"}";

	string binsTemplate = this->dataType + " bin<%BNUM%> = convert_" + this->dataType + "(0);\n";
	string countersTemplate = "unsigned int counter<%BNUM%> = 0;\n";
	string accumulateTemplate = "bin<%BNUM%> += cSample * (bin == <%BNUM%>);\n"
		"counter<%BNUM%> += 1 * (bin == <%BNUM%>);\n";
	string storeTemplate = "globalItem = (((dm * " + nrBins_s + ") + <%BNUM%>) * " + nrPaddedPeriods_s + ") + periodIndex;\n"
		"pCounter = counters[globalItem];\n"
		"pValue = bins[globalItem];\n"
		"addedFraction = convert_float(counter<%BNUM%>) / ( ( counter<%BNUM%> + pCounter ) + ( 1 * ( counter<%BNUM%> == 0 ) ) );\n"
		"bin<%BNUM%> /= counter<%BNUM%> * ( 1 * ( counter<%BNUM%> > 0 ) );\n"
		"counters[globalItem] = pCounter + counter<%BNUM%>;\n"
		"bins[globalItem] = (addedFraction * bin<%BNUM%>) + ((1.0f - addedFraction) * pValue);\n";
	// End kernel's template
	
	string * bins = new string();
	string * counters = new string();
	string * accumulates = new string();
	string * stores = new string();
	
	for ( unsigned int bin = 0; bin < observation->getNrBins(); bin++ ) {
		string bin_s = toStringValue< unsigned int >(bin);
		string * temp = 0;

		temp = replace(&binsTemplate, "<%BNUM%>", bin_s);
		bins->append(*temp);
		delete temp;
		temp = replace(&countersTemplate, "<%BNUM%>", bin_s);
		counters->append(*temp);
		delete temp;
		temp = replace(&accumulateTemplate, "<%BNUM%>", bin_s);
		accumulates->append(*temp);
		delete temp;
		temp = replace(&storeTemplate, "<%BNUM%>", bin_s);
		stores->append(*temp);
		delete temp;
	}
	this->code = replace(this->code, "<%BINS%>", *bins, true);
	delete bins;
	this->code = replace(this->code, "<%COUNTERS%>", *counters, true);
	delete counters;
	this->code = replace(this->code, "<%ACCUMULATE%>", *accumulates, true);
	delete accumulates;
	this->code = replace(this->code, "<%STORE%>", *stores, true);
	delete stores;

	globalSize = cl::NDRange(observation->getNrPeriods(), observation->getNrDMs());
	localSize = cl::NDRange(nrPeriodsPerBlock, 1);

	this->gb = giga(static_cast< long long unsigned int >(observation->getNrDMs()) * observation->getNrPeriods() * ( ( ( observation->getNrSamplesPerSecond() / nrPeriodsPerBlock ) * sizeof(T) ) + ( ( 2 * sizeof(T) ) + ( 2 * sizeof(unsigned int) ) ) ));
	this->gflop = giga(static_cast< long long unsigned int >(observation->getNrDMs()) * observation->getNrPeriods() * observation->getNrSamplesPerSecond() * observation->getNrBins());
	this->arInt = this->gflop / this->gb;

	this->compile();
}

template< typename T > void Folding< T >::operator()(unsigned int second, CLData< T > * input, CLData< T > * output, CLData< unsigned int > * counters) throw (OpenCLError) {
	this->setArgument(0, second);
	this->setArgument(1, *(input->getDeviceData()));
	this->setArgument(2, *(output->getDeviceData()));
	this->setArgument(3, *(counters->getDeviceData()));

	this->run(globalSize, localSize);
}

template< typename T > inline void Folding< T >::setNrPeriodsPerBlock(unsigned int periods) {
	nrPeriodsPerBlock = periods;
}

template< typename T > inline void Folding< T >::setObservation(Observation< T > * obs) {
	observation = obs;
}

} // PulsarSearch

#endif // FOLDING_HPP
