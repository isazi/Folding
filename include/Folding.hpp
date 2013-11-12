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
using isa::Exceptions::OpenCLError;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <utils.hpp>
using isa::utils::giga;
using isa::utils::toStringValue;
using isa::utils::toString;
using isa::utils::replace;
#include <Kernel.hpp>
using isa::OpenCL::Kernel;
#include <Observation.hpp>
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

	inline void setNrDMsPerThread(unsigned int DMs);
	inline void setNrPeriodsPerThread(unsigned int periods);
	inline void setNrBinsPerThread(unsigned int bins);

	inline void setObservation(Observation< T > * obs);

private:
	unsigned int nrDMsPerBlock;
	unsigned int nrPeriodsPerBlock;
	unsigned int nrBinsPerBlock;
	unsigned int nrDMsPerThread;
	unsigned int nrPeriodsPerThread;
	unsigned int nrBinsPerThread;
	cl::NDRange globalSize;
	cl::NDRange localSize;

	Observation< T > * observation;
};


// Implementation
template< typename T > Folding< T >::Folding(string name, string dataType) : Kernel< T >(name, dataType), nrDMsPerBlock(0), nrPeriodsPerBlock(0), nrBinsPerBlock(0), nrDMsPerThread(0), nrPeriodsPerThread(0), nrBinsPerThread(0), globalSize(cl::NDRange(1, 1, 1)), localSize(cl::NDRange(1, 1, 1)), observation(0) {}

template< typename T > void Folding< T >::generateCode() throw (OpenCLError) {
	// Begin kernel's template
	string nrSamplesPerSecond_s = toStringValue< unsigned int >(observation->getNrSamplesPerSecond());
	string nrPaddedDMs_s  = toStringValue< unsigned int >(observation->getNrPaddedDMs());
	string nrPeriods_s = toStringValue< unsigned int >(observation->getNrPeriods());
	string firstPeriod_s = toStringValue< unsigned int >(observation->getFirstPeriod());
	string periodStep_s = toStringValue< unsigned int >(observation->getPeriodStep());
	string nrBins_s = toStringValue< unsigned int >(observation->getNrBins());
	string nrDMsPerBlock_s = toStringValue< unsigned int >(nrDMsPerBlock);
	string nrDMsPerThread_s = toStringValue< unsigned int >(nrDMsPerThread);
	string nrPeriodsPerBlock_s = toStringValue< unsigned int >(nrPeriodsPerBlock);
	string nrPeriodsPerThread_s = toStringValue< unsigned int >(nrPeriodsPerThread);
	string nrBinsPerBlock_s = toStringValue< unsigned int >(nrBinsPerBlock);
	string nrBinsPerThread_s = toStringValue< unsigned int >(nrBinsPerThread);

	delete this->code;
	this->code = new string();
	*(this->code) = "__kernel void " + this->name + "(__global const " + this->dataType + " * const restrict samples, __global " + this->dataType + " * const restrict bins, __global unsigned int * const restrict counters) {\n"
	"<%DEFS%>"
	"<%LOADS%>"
	"\n"
	"unsigned int sample = 0;"
	"<%COMPUTE%>"
	"\n"
	"<%STORE%>"
	"}\n";

	string defsDMTemplate = "const unsigned int DM<%DM_NUM%> = (get_group_id(0) * " + nrDMsPerBlock_s + " * " + nrDMsPerThread_s + ") + get_local_id(0) + (<%DM_NUM%> * " + nrDMsPerBlock_s + ");\n";

	string defsPeriodTemplate = "const unsigned int period<%PERIOD_NUM%> = (get_group_id(1) * " + nrPeriodsPerBlock_s + " * " + nrPeriodsPerThread_s+  ") + get_local_id(1) + (<%PERIOD_NUM%> * " + nrPeriodsPerBlock_s + ");\n"
		"const unsigned int period<%PERIOD_NUM%>Value = " + firstPeriod_s + " + (period<%PERIOD_NUM%> * " + periodStep_s + ");\n"
		"const unsigned int samplesPerBin<%PERIOD_NUM%> = period<%PERIOD_NUM%>Value / " + firstPeriod_s + ";\n";

	string defsBinTemplate = "const unsigned int bin<%BIN_NUM%> = (get_group_id(2) * " + nrBinsPerBlock_s + " * " + nrBinsPerThread_s + ") + get_local_id(2) + (<%BIN_NUM%> * " + nrBinsPerBlock_s + ");\n";

	string defsTemplate = "unsigned int foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> = 0;\n"
		+ this->dataType + " foldedSampleDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> = 0;\n"
		"unsigned int pCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> = 0;\n";
		
	string loadsTemplate = "pCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> = counters[(bin<%BIN_NUM%> * " + nrPeriods_s + " * " + nrPaddedDMs_s + ") + (period<%PERIOD_NUM%> * " + nrPaddedDMs_s + ") + DM<%DM_NUM%>];\n";

	string computeTemplate = "sample = (bin<%BIN_NUM%> * period<%PERIOD_NUM%>) + bin<%BIN_NUM%> + ((pCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> / samplesPerBin<%PERIOD_NUM%>) * period<%PERIOD_NUM%>Value) + (pCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> % samplesPerBin<%PERIOD_NUM%>);\n"
	"if ( (sample % "+ nrSamplesPerSecond_s + ") == 0 ) {\n"
	"sample = 0;\n"
	"} else {\n"
	"sample = (sample % "+ nrSamplesPerSecond_s + ") - (sample / "+ nrSamplesPerSecond_s + ");\n"
	"}\n"
	"while ( sample < " + nrSamplesPerSecond_s + " ) {\n"
	"foldedSampleDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> += samples[(sample * " + nrPaddedDMs_s + ") + DM<%DM_NUM%>];\n"
	"foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>++;\n"
	"if ( ((foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> + pCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>) % samplesPerBin<%PERIOD_NUM%>) == 0 ) {\n"
	"sample += period<%PERIOD_NUM%>Value;\n"
	"} else {\n"
	"sample++;\n"
	"}\n"
	"}\n";

	string storeTemplate = "if ( foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> > 0 ) {\n"
	"const unsigned int outputItem = (bin<%BIN_NUM%> * " + nrPeriods_s + " * " + nrPaddedDMs_s + ") + (period<%PERIOD_NUM%> * " + nrPaddedDMs_s + ") + DM<%DM_NUM%>;\n"
	"const "+ this->dataType + " pValue = bins[outputItem];\n"
	"float addedFraction = convert_float(foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>) / (foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> + pCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>);\n"
	"foldedSampleDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> /= foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>;\n"
	"counters[outputItem] = pCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> + foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>;\n"
	"bins[outputItem] = (addedFraction * foldedSampleDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>) + ((1.0f - addedFraction) * pValue);\n"
	"}\n";
	// End kernel's template

	string * defs = new string();
	string * loads = new string();
	string * computes = new string();
	string * stores = new string();
	for ( unsigned int DM = 0; DM < nrDMsPerThread; DM++ ) {
		string * DM_s = toString< unsigned int >(DM);
		string * temp = 0;

		temp = replace(&defsDMTemplate, "<%DM_NUM%>", *DM_s);
		defs->append(*temp);
		delete temp;

		delete DM_s;
	}
	for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
		string * period_s = toString< unsigned int >(period);
		string * temp = 0;

		temp = replace(&defsPeriodTemplate, "<%PERIOD_NUM%>", *period_s);
		defs->append(*temp);
		delete temp;

		delete period_s;
	}
	for ( unsigned int bin = 0; bin < nrBinsPerThread; bin++ ) {
		string * bin_s = toString< unsigned int >(bin);
		string * temp = 0;

		temp = replace(&defsBinTemplate, "<%BIN_NUM%>", *bin_s);
		defs->append(*temp);
		delete temp;

		delete bin_s;
	}
	for ( unsigned int bin = 0; bin < nrBinsPerThread; bin++ ) {
		string * bin_s = toString< unsigned int >(bin);
		
		for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
			string * period_s = toString< unsigned int >(period);
			
			for ( unsigned int DM = 0; DM < nrDMsPerThread; DM++ ) {
				string * DM_s = toString< unsigned int >(DM);
				string * temp = 0;

				temp = replace(&defsTemplate, "<%BIN_NUM%>", *bin_s);
				temp = replace(temp, "<%PERIOD_NUM%>", *period_s, true);
				temp = replace(temp, "<%DM_NUM%>", *DM_s, true);
				defs->append(*temp);
				delete temp;

				temp = replace(&loadsTemplate, "<%BIN_NUM%>", *bin_s);
				temp = replace(temp, "<%PERIOD_NUM%>", *period_s, true);
				temp = replace(temp, "<%DM_NUM%>", *DM_s, true);
				loads->append(*temp);
				delete temp;

				temp = replace(&storeTemplate, "<%BIN_NUM%>", *bin_s);
				temp = replace(temp, "<%PERIOD_NUM%>", *period_s, true);
				temp = replace(temp, "<%DM_NUM%>", *DM_s, true);
				stores->append(*temp);
				delete temp;

				delete DM_s;		
			}

			delete period_s;
		}

		delete bin_s;
	}
	for ( unsigned int DM = 0; DM < nrDMsPerThread; DM++ ) {
		string * DM_s = toString< unsigned int >(DM);

		for ( unsigned int bin = 0; bin < nrBinsPerThread; bin++ ) {
				string * bin_s = toString< unsigned int >(bin);

				for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
				string * period_s = toString< unsigned int >(period);
				string * temp = 0;

				temp = replace(&computeTemplate, "<%BIN_NUM%>", *bin_s);
				temp = replace(temp, "<%PERIOD_NUM%>", *period_s, true);
				temp = replace(temp, "<%DM_NUM%>", *DM_s, true);
				computes->append(*temp);
				delete temp;

				delete period_s;
			}

			delete bin_s;
		}

		delete DM_s;
	}
	this->code = replace(this->code, "<%DEFS%>", *defs, true);
	this->code = replace(this->code, "<%LOADS%>", *loads, true);
	this->code = replace(this->code, "<%COMPUTE%>", *computes, true);
	this->code = replace(this->code, "<%STORE%>", *stores, true);
	delete defs;
	delete computes;
	delete stores;

	globalSize = cl::NDRange(observation->getNrDMs() / nrDMsPerThread, observation->getNrPeriods() / nrPeriodsPerThread, observation->getNrBins() / nrBinsPerThread);
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

template< typename T > inline void Folding< T >::setNrDMsPerThread(unsigned int DMs) {
	nrDMsPerThread = DMs;
}

template< typename T > inline void Folding< T >::setNrPeriodsPerThread(unsigned int periods) {
	nrPeriodsPerThread = periods;
}

template< typename T > inline void Folding< T >::setNrBinsPerThread(unsigned int bins) {
	nrBinsPerThread = bins;
}

template< typename T > inline void Folding< T >::setObservation(Observation< T > * obs) {
	observation = obs;
}

} // PulsarSearch

#endif // FOLDING_HPP
