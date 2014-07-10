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

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>

#include <Exceptions.hpp>
#include <CLData.hpp>
#include <utils.hpp>
#include <Kernel.hpp>
#include <Observation.hpp>


#ifndef FOLDING_HPP
#define FOLDING_HPP

namespace PulsarSearch {

// OpenCL folding algorithm
template< typename T > class Folding : public isa::OpenCL::Kernel< T > {
public:
	Folding(std::string name, std::string dataType);

	void generateCode() throw (isa::Exceptions::OpenCLError);
	void operator()(unsigned int second, isa::OpenCL::CLData< T > * input, isa::OpenCL::CLData< T > * output, isa::OpenCL::CLData< unsigned int > * readCounters, isa::OpenCL::CLData< unsigned int > * writeCounters) throw (isa::Exceptions::OpenCLError);

	inline void setNrDMsPerBlock(unsigned int DMs);
	inline void setNrPeriodsPerBlock(unsigned int periods);
	inline void setNrBinsPerBlock(unsigned int bins);

	inline void setNrDMsPerThread(unsigned int DMs);
	inline void setNrPeriodsPerThread(unsigned int periods);
	inline void setNrBinsPerThread(unsigned int bins);

	inline void setObservation(AstroData::Observation< T > * obs);
	inline void setNrSamplesPerBin(isa::OpenCL::CLData< unsigned int > * samplesPerBin);

private:
	unsigned int nrDMsPerBlock;
	unsigned int nrPeriodsPerBlock;
	unsigned int nrBinsPerBlock;
	unsigned int nrDMsPerThread;
	unsigned int nrPeriodsPerThread;
	unsigned int nrBinsPerThread;
	cl::NDRange globalSize;
	cl::NDRange localSize;

	AstroData::Observation< T > * observation;
	isa::OpenCL::CLData< unsigned int > * nrSamplesPerBin;
};


// Implementation
template< typename T > Folding< T >::Folding(std::string name, std::string dataType) : isa::OpenCL::Kernel< T >(name, dataType), nrDMsPerBlock(0), nrPeriodsPerBlock(0), nrBinsPerBlock(0), nrDMsPerThread(0), nrPeriodsPerThread(0), nrBinsPerThread(0), globalSize(cl::NDRange(1, 1, 1)), localSize(cl::NDRange(1, 1, 1)), observation(0), nrSamplesPerBin(0) {}

template< typename T > void Folding< T >::generateCode() throw (isa::Exceptions::OpenCLError) {
	// Begin kernel's template
	std::string nrSamplesPerSecond_s = isa::utils::toString< unsigned int >(observation->getNrSamplesPerSecond());
	std::string nrPaddedDMs_s  = isa::utils::toString< unsigned int >(observation->getNrPaddedDMs());
	std::string nrPeriods_s = isa::utils::toString< unsigned int >(observation->getNrPeriods());
	std::string firstPeriod_s = isa::utils::toString< unsigned int >(observation->getFirstPeriod());
	std::string periodStep_s = isa::utils::toString< unsigned int >(observation->getPeriodStep());
	std::string nrPaddedBins_s = isa::utils::toString< unsigned int >(observation->getNrPaddedBins());
	std::string nrDMsPerBlock_s = isa::utils::toString< unsigned int >(nrDMsPerBlock);
	std::string nrDMsPerThread_s = isa::utils::toString< unsigned int >(nrDMsPerThread);
	std::string nrPeriodsPerBlock_s = isa::utils::toString< unsigned int >(nrPeriodsPerBlock);
	std::string nrPeriodsPerThread_s = isa::utils::toString< unsigned int >(nrPeriodsPerThread);
	std::string nrBinsPerBlock_s = isa::utils::toString< unsigned int >(nrBinsPerBlock);
	std::string nrBinsPerThread_s = isa::utils::toString< unsigned int >(nrBinsPerThread);

	delete this->code;
	this->code = new std::string();
	*(this->code) = "__kernel void " + this->name + "(const unsigned int second, __global const " + this->dataType + " * const restrict samples, __global " + this->dataType + " * const restrict bins, __global const unsigned int * const restrict readCounters, __global unsigned int * const restrict writeCounters, __global const unsigned int * const restrict nrSamplesPerBin) {\n"
    "<%DEFS%>"
    "\n"
    "unsigned int sample = 0;"
    "<%COMPUTE%>"
    "\n"
    "<%STORE%>"
    "}\n";

	std::string defsDMTemplate = "const unsigned int DM<%DM_NUM%> = (get_group_id(0) * " + nrDMsPerBlock_s + " * " + nrDMsPerThread_s + ") + get_local_id(0) + (<%DM_NUM%> * " + nrDMsPerBlock_s + ");\n";

	std::string defsPeriodTemplate = "const unsigned int period<%PERIOD_NUM%> = (get_group_id(1) * " + nrPeriodsPerBlock_s + " * " + nrPeriodsPerThread_s+  ") + get_local_id(1) + (<%PERIOD_NUM%> * " + nrPeriodsPerBlock_s + ");\n"
      "const unsigned int period<%PERIOD_NUM%>Value = " + firstPeriod_s + " + (period<%PERIOD_NUM%> * " + periodStep_s + ");\n";

	std::string defsBinTemplate = "const unsigned int bin<%BIN_NUM%> = (get_group_id(2) * " + nrBinsPerBlock_s + " * " + nrBinsPerThread_s + ") + get_local_id(2) + (<%BIN_NUM%> * " + nrBinsPerBlock_s + ");\n";

	std::string samplesPerBinTemplate = "const unsigned int samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%> = nrSamplesPerBin[(period<%PERIOD_NUM%> * " + isa::utils::toString(observation->getNrBins() * isa::utils::pad(2, observation->getPadding())) + ") + (bin<%BIN_NUM%> * " + isa::utils::toString(isa::utils::pad(2, observation->getPadding())) + ")];\n"
		"const unsigned int offsetp<%PERIOD_NUM%>b<%BIN_NUM%> = nrSamplesPerBin[(period<%PERIOD_NUM%> * " + isa::utils::toString(observation->getNrBins() * isa::utils::pad(2, observation->getPadding())) + ") + (bin<%BIN_NUM%> * " + isa::utils::toString(isa::utils::pad(2, observation->getPadding())) + ") + 1];\n"
		"const unsigned int pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> = readCounters[(period<%PERIOD_NUM%> * " + nrPaddedBins_s + ") + bin<%BIN_NUM%>];\n";

	std::string defsTemplate = "unsigned int foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> = 0;\n"
		+ this->dataType + " foldedSampleDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> = 0;\n";

	std::string computeTemplate = "if ( samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%> > 0 ) {\n"
    "sample = offsetp<%PERIOD_NUM%>b<%BIN_NUM%> + ((pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> / samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%>) * period<%PERIOD_NUM%>Value) + (pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> % samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%>);\n"
    "if ( (sample / "+ nrSamplesPerSecond_s + ") == second ) {\n"
    "sample %= "+ nrSamplesPerSecond_s + ";\n"
    "}\n"
    "while ( sample < " + nrSamplesPerSecond_s + " ) {\n"
    "foldedSampleDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> += samples[(sample * " + nrPaddedDMs_s + ") + DM<%DM_NUM%>];\n"
    "foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>++;\n"
    "if ( ((foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> + pCounterp<%PERIOD_NUM%>b<%BIN_NUM%>) % samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%>) == 0 ) {\n"
    "sample += period<%PERIOD_NUM%>Value - (samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%> - 1);\n"
    "} else {\n"
    "sample++;\n"
    "}\n"
    "}\n"
    "}\n";

	std::string storeTemplate = "if ( foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> > 0 ) {\n"
    "const unsigned int outputItem = (bin<%BIN_NUM%> * " + nrPeriods_s + " * " + nrPaddedDMs_s + ") + (period<%PERIOD_NUM%> * " + nrPaddedDMs_s + ") + DM<%DM_NUM%>;\n"
    "const "+ this->dataType + " pValue = bins[outputItem];\n"
    "float addedFraction = convert_float(foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>) / (foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> + pCounterp<%PERIOD_NUM%>b<%BIN_NUM%>);\n"
    "foldedSampleDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> /= foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>;\n"
    "writeCounters[(period<%PERIOD_NUM%> * " + nrPaddedBins_s + ") + bin<%BIN_NUM%>] = pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> + foldedCounterDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>;\n"
    "bins[outputItem] = (addedFraction * foldedSampleDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%>) + ((1.0f - addedFraction) * pValue);\n"
    "}\n";
	// End kernel's template

	std::string * defs = new std::string();
	std::string * computes = new std::string();
	std::string * stores = new std::string();
	for ( unsigned int DM = 0; DM < nrDMsPerThread; DM++ ) {
		std::string DM_s = isa::utils::toString< unsigned int >(DM);
		std::string * temp = 0;

		temp = isa::utils::replace(&defsDMTemplate, "<%DM_NUM%>", DM_s);
		defs->append(*temp);
		delete temp;
	}
	for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
		std::string period_s = isa::utils::toString< unsigned int >(period);
		std::string * temp = 0;

		temp = isa::utils::replace(&defsPeriodTemplate, "<%PERIOD_NUM%>", period_s);
		defs->append(*temp);
		delete temp;
	}
	for ( unsigned int bin = 0; bin < nrBinsPerThread; bin++ ) {
		std::string bin_s = isa::utils::toString< unsigned int >(bin);
		std::string * temp = 0;

		temp = isa::utils::replace(&defsBinTemplate, "<%BIN_NUM%>", bin_s);
		defs->append(*temp);
		delete temp;
	}
	for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
		std::string period_s = isa::utils::toString< unsigned int >(period);

		for ( unsigned int bin = 0; bin < nrBinsPerThread; bin++ ) {
      std::string bin_s = isa::utils::toString< unsigned int >(bin);
			std::string * temp = 0;

			temp = isa::utils::replace(&samplesPerBinTemplate, "<%BIN_NUM%>", bin_s);
			temp = isa::utils::replace(temp, "<%PERIOD_NUM%>", period_s, true);
			defs->append(*temp);
			delete temp;
		}
	}
	for ( unsigned int bin = 0; bin < nrBinsPerThread; bin++ ) {
    std::string bin_s = isa::utils::toString< unsigned int >(bin);

		for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
      std::string period_s = isa::utils::toString< unsigned int >(period);

			for ( unsigned int DM = 0; DM < nrDMsPerThread; DM++ ) {
        std::string DM_s = isa::utils::toString< unsigned int >(DM);
				std::string * temp = 0;

				temp = isa::utils::replace(&defsTemplate, "<%BIN_NUM%>", bin_s);
				temp = isa::utils::replace(temp, "<%PERIOD_NUM%>", period_s, true);
				temp = isa::utils::replace(temp, "<%DM_NUM%>", DM_s, true);
				defs->append(*temp);
				delete temp;

				temp = isa::utils::replace(&storeTemplate, "<%BIN_NUM%>", bin_s);
				temp = isa::utils::replace(temp, "<%PERIOD_NUM%>", period_s, true);
				temp = isa::utils::replace(temp, "<%DM_NUM%>", DM_s, true);
				stores->append(*temp);
				delete temp;
			}
		}
	}
	for ( unsigned int DM = 0; DM < nrDMsPerThread; DM++ ) {
    std::string DM_s = isa::utils::toString< unsigned int >(DM);

		for ( unsigned int bin = 0; bin < nrBinsPerThread; bin++ ) {
        std::string bin_s = isa::utils::toString< unsigned int >(bin);

				for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
        std::string period_s = isa::utils::toString< unsigned int >(period);
				std::string * temp = 0;

				temp = isa::utils::replace(&computeTemplate, "<%BIN_NUM%>", bin_s);
				temp = isa::utils::replace(temp, "<%PERIOD_NUM%>", period_s, true);
				temp = isa::utils::replace(temp, "<%DM_NUM%>", DM_s, true);
				computes->append(*temp);
				delete temp;
			}
		}
	}
	this->code = isa::utils::replace(this->code, "<%DEFS%>", *defs, true);
	this->code = isa::utils::replace(this->code, "<%COMPUTE%>", *computes, true);
	this->code = isa::utils::replace(this->code, "<%STORE%>", *stores, true);
	delete defs;
	delete computes;
	delete stores;

	globalSize = cl::NDRange(observation->getNrPaddedDMs() / nrDMsPerThread, observation->getNrPeriods() / nrPeriodsPerThread, observation->getNrBins() / nrBinsPerThread);
	localSize = cl::NDRange(nrDMsPerBlock, nrPeriodsPerBlock, nrBinsPerBlock);

	this->gflop = isa::utils::giga(static_cast< long long unsigned int >(observation->getNrDMs()) * observation->getNrPeriods() * observation->getNrSamplesPerSecond());

	this->compile();
}

template< typename T > void Folding< T >::operator()(unsigned int second, isa::OpenCL::CLData< T > * input, isa::OpenCL::CLData< T > * output, isa::OpenCL::CLData< unsigned int > * readCounters, isa::OpenCL::CLData< unsigned int > * writeCounters) throw (isa::Exceptions::OpenCLError) {
	this->setArgument(0, second);
	this->setArgument(1, *(input->getDeviceData()));
	this->setArgument(2, *(output->getDeviceData()));
	this->setArgument(3, *(readCounters->getDeviceData()));
	this->setArgument(4, *(writeCounters->getDeviceData()));
	this->setArgument(5, *(nrSamplesPerBin->getDeviceData()));

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

template< typename T > inline void Folding< T >::setObservation(AstroData::Observation< T > * obs) {
	observation = obs;
}

template< typename T > inline void Folding< T >::setNrSamplesPerBin(isa::OpenCL::CLData< unsigned int > * samplesPerBin) {
	nrSamplesPerBin = samplesPerBin;
}

} // PulsarSearch

#endif // FOLDING_HPP
