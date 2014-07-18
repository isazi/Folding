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

#include <string>
#include <vector>
#include <x86intrin.h>

#include <utils.hpp>
#include <Observation.hpp>


#ifndef FOLDING_HPP
#define FOLDING_HPP

namespace PulsarSearch {

// Sequential folding
template< typename T > void folding(const unsigned int second, const Observation< T > & observation, const std::vector< T > & samples, std::vector< T > & bins, std::vector< unsigned int > & counters);
// OpenCL folding algorithm
template< typename T > std::string * getFoldingOpenCL(const unsigned int nrDMsPerBlock, const unsigned int nrPeriodsPerBlock, const unsigned int nrBinsPerBlock, const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, const unsigned int nrBinsPerThread, std::string & dataType, const AstroData::Observation< T > & observation);


// Implementations
template< typename T > void folding(const unsigned int second, const Observation< T > & observation, const std::vector< T > & samples, std::vector< T > & bins, std::vector< unsigned int > & counters) {
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int periodIndex = 0; periodIndex < observation.getNrPeriods(); periodIndex++ ) {
      const unsigned int periodValue = observation.getFirstPeriod() + (periodIndex * observation.getPeriodStep());

      for ( unsigned int globalSample = 0; globalSample < observation.getNrSamplesPerSecond(); globalSample++ ) {
        const unsigned int sample = (second * observation.getNrSamplesPerSecond()) + globalSample;
        const float phase = (sample / static_cast< float >(periodValue)) - (sample / periodValue);
        const unsigned int bin = static_cast< unsigned int >(phase * static_cast< float >(observation.getNrBins()));
        const unsigned int globalItem = (((dm * observation.getNrPeriods()) + periodIndex) * observation.getNrPaddedBins()) + bin;

        const T pValue = bins[globalItem];
        T cValue = samples[(dm * observation.getNrSamplesPerPaddedSecond()) + globalSample];
        const unsigned int pCounter = counters[globalItem];
        unsigned int cCounter = pCounter + 1;

        if ( pCounter != 0 ) {
          cValue = pValue + ((cValue - pValue) / cCounter);
        }
        bins[globalItem] = cValue;
        counters[globalItem] = cCounter;
      }
    }
  }
}

template< typename T > std::string * getFoldingOpenCL(const unsigned int nrDMsPerBlock, const unsigned int nrPeriodsPerBlock, const unsigned int nrBinsPerBlock, const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, const unsigned int nrBinsPerThread, std::string & dataType, const AstroData::Observation< T > & observation) {
  std::string * code = new std::string();
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

	// Begin kernel's template
	*code = "__kernel void folding(const unsigned int second, __global const " + dataType + " * const restrict samples, __global " + dataType + " * const restrict bins, __global const unsigned int * const restrict readCounters, __global unsigned int * const restrict writeCounters, __global const unsigned int * const restrict nrSamplesPerBin) {\n"
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
		+ dataType + " foldedSampleDM<%DM_NUM%>p<%PERIOD_NUM%>b<%BIN_NUM%> = 0;\n";

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
    "const "+ dataType + " pValue = bins[outputItem];\n"
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
	code = isa::utils::replace(code, "<%DEFS%>", *defs, true);
	code = isa::utils::replace(code, "<%COMPUTE%>", *computes, true);
	code = isa::utils::replace(code, "<%STORE%>", *stores, true);
	delete defs;
	delete computes;
	delete stores;

  return code;
}

} // PulsarSearch

#endif // FOLDING_HPP
