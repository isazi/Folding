// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
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
#include <map>
#include <x86intrin.h>

#include <utils.hpp>
#include <Observation.hpp>


#ifndef FOLDING_HPP
#define FOLDING_HPP

namespace PulsarSearch {

template< typename T > using foldingFunc = void (*)(unsigned int, AstroData::Observation &, float *, float *, unsigned int *, unsigned int *, unsigned int *);

// Sequential folding
template< typename T > void folding(const unsigned int second, const AstroData::Observation & observation, const std::vector< T > & samples, std::vector< T > & bins, std::vector< unsigned int > & counters);
// AVX folding algorithm
std::string * getFoldingAVX(const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, const unsigned int nrBinsPerThread);
// Phi folding algorithm
std::string * getFoldingPhi(const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, const unsigned int nrBinsPerThread);


// Implementations
template< typename T > void folding(const unsigned int second, const AstroData::Observation & observation, const std::vector< T > & samples, std::vector< T > & bins, std::vector< unsigned int > & counters) {
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int periodIndex = 0; periodIndex < observation.getNrPeriods(); periodIndex++ ) {
      const unsigned int periodValue = observation.getFirstPeriod() + (periodIndex * observation.getPeriodStep());

      for ( unsigned int globalSample = 0; globalSample < observation.getNrSamplesPerSecond(); globalSample++ ) {
        const unsigned int sample = (second * observation.getNrSamplesPerSecond()) + globalSample;
        const float phase = (sample / static_cast< float >(periodValue)) - (sample / periodValue);
        const unsigned int bin = static_cast< unsigned int >(phase * static_cast< float >(observation.getNrBins()));
        const unsigned int globalItem = (dm * observation.getNrPeriods() * observation.getNrPaddedBins()) + (periodIndex * observation.getNrPaddedBins()) + bin;

        const T pValue = bins[globalItem];
        const unsigned int pCounter = counters[globalItem];
        T cValue = samples[(dm * observation.getNrSamplesPerPaddedSecond()) + globalSample];
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

std::string * getFoldingAVX(const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, const unsigned int nrBinsPerThread) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "namespace PulsarSearch {\n"
    "template< typename T > void foldingAVX" + isa::utils::toString< unsigned int >(nrDMsPerThread) + "x" + isa::utils::toString< unsigned int >(nrPeriodsPerThread) + "x" + isa::utils::toString< unsigned int >(nrBinsPerThread) + "(const unsigned int second, const AstroData::Observation & observation, const float * const __restrict__ samples, float * const __restrict__ bins, const unsigned int * const __restrict__ readCounters, unsigned int * const __restrict__ writeCounters, const unsigned int * const __restrict__ samplesPerBin) {\n"
    "#pragma omp parallel for schedule(static)\n"
    "for ( unsigned int periodIndex = 0; periodIndex < observation.getNrPeriods(); periodIndex += " + isa::utils::toString< unsigned int >(nrPeriodsPerThread) + ") {\n"
    "<%PERIOD_VARS%>"
    "\n"
    "#pragma omp parallel for schedule(static)\n"
		"for ( unsigned int bin = 0; bin < observation.getNrBins(); bin += " + isa::utils::toString< unsigned int >(nrBinsPerThread) + ") {\n"
    "<%BIN_VARS%>"
    "\n"
    "#pragma omp parallel for schedule(static)\n"
    "for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm += 8 * " + isa::utils::toString< unsigned int >(nrDMsPerThread) + ") {\n"
    "<%DM_VARS%>"
    "\n"
    "<%COMPUTE%>"
    "}\n"
    "}\n"
    "}\n"
    "}\n"
    "}\n";
  std::string periodVarsTemplate = "const unsigned int periodValuep<%PERIOD_NUM%> = observation.getFirstPeriod() + ((periodIndex + <%PERIOD_NUM%>) * observation.getPeriodStep());\n";
  std::string binVarsTemplate = "const unsigned int pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> = readCounters[((periodIndex + <%PERIOD_NUM%>) * observation.getNrPaddedBins()) + (bin + <%BIN_NUM%>)];\n"
    "unsigned int foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>;\n"
    "unsigned int samplep<%PERIOD_NUM%>b<%BIN_NUM%>;\n";
  std::string dmVarsTemplate = "__m256 foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%> = _mm256_setzero_ps();\n";
  std::string computeTemplate = "foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%> = 0;\n"
    "samplep<%PERIOD_NUM%>b<%BIN_NUM%> = samplesPerBin->at(((periodIndex + <%PERIOD_NUM%>) * 2 * observation.getNrPaddedBins()) + ((bin + <%BIN_NUM%>) * 2) + 1) + ((pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> / samplesPerBin->at(((periodIndex + <%PERIOD_NUM%>) * 2 * observation.getNrPaddedBins()) + ((bin + <%BIN_NUM%>) * 2))) * periodValuep<%PERIOD_NUM%>) + (pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> % samplesPerBin->at(((periodIndex + <%PERIOD_NUM%>) * 2 * observation.getNrPaddedBins()) + ((bin + <%BIN_NUM%>) * 2)));\n"
    "\n"
    "if ( (samplep<%PERIOD_NUM%>b<%BIN_NUM%> / observation.getNrSamplesPerSecond()) == second ) {\n"
    "samplep<%PERIOD_NUM%>b<%BIN_NUM%> %= observation.getNrSamplesPerSecond();\n"
    "}\n"
    "while ( samplep<%PERIOD_NUM%>b<%BIN_NUM%> < observation.getNrSamplesPerSecond() ) {\n"
    "foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%> = _mm256_add_ps(foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%>, _mm256_load_ps(&(samples[(samplep<%PERIOD_NUM%>b<%BIN_NUM%> * observation.getNrPaddedDMs()) + (dm + <%DM_NUM%>)])));\n"
    "foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>++;\n"
    "\n"
    "if ( (foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%> + pCounterp<%PERIOD_NUM%>b<%BIN_NUM%>) % samplesPerBin->at(((periodIndex + <%PERIOD_NUM%>) * 2 * observation.getNrPaddedBins()) + ((bin + <%BIN_NUM%>) * 2)) == 0 ) {\n"
    "samplep<%PERIOD_NUM%>b<%BIN_NUM%> += periodValuep<%PERIOD_NUM%>;\n"
    "} else {\n"
    "samplep<%PERIOD_NUM%>b<%BIN_NUM%>++;\n"
    "}\n"
    "}\n"
    "\n"
    "if ( foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%> > 0 ) {\n"
    "const __m256 pValue = _mm256_load_ps(&(bins[((bin + <%BIN_NUM%>) * observation.getNrPeriods() * observation.getNrPaddedDMs()) + ((periodIndex + <%PERIOD_NUM%>) * observation.getNrPaddedDMs()) + (dm + <%DM_NUM%>)]));\n"
    "const __m256 cValue = _mm256_div_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>), pValue), foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%>), _mm256_add_ps(_mm256_set1_ps(pCounterp<%PERIOD_NUM%>n<%BIN_NUM%>), _mm256_set1_ps(foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>)));\n"
    "_mm256_store_ps(&((bin + <%BIN_NUM%>)s[((bin + <%BIN_NUM%>) * observation.getNrPeriods() * observation.getNrPaddedDMs()) + ((periodIndex + <%PERIOD_NUM%>) * observation.getNrPaddedDMs()) + (dm + <%DM_NUM%>)]), cValue)\n"
    "writeCounters[((periodIndex + <%PERIOD_NUM%>) * observation.getNrPaddedBins()) + (bin + <%BIN_NUM%>)] = pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> + foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>;\n"
    "}\n";
  // End kernel's template
  
  std::string * periodVars_s = new std::string();
  std::string * binVars_s = new std::string();
  std::string * dmVars_s = new std::string();
  std::string * compute_s = new std::string();

  for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
    std::string period_s = isa::utils::toString< unsigned int >(period);
    std::string * temp = 0;

    temp = isa::utils::replace(&periodVarsTemplate, "<%PERIOD_NUM%>", period_s);
    periodVars_s->append(*temp);
    delete temp;

    for ( unsigned int bin = 0; bin < nrBinsPerThread; bin++ ) {
      std::string bin_s = isa::utils::toString< unsigned int >(bin);
      std::string * temp = 0;

      temp = isa::utils::replace(&binVarsTemplate, "<%PERIOD_NUM%>", period_s);
      temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
      binVars_s->append(*temp);
      delete temp;

      for ( unsigned int dm = 0; dm < nrDMsPerThread; dm++ ) {
        std::string dm_s = isa::utils::toString< unsigned int >(dm);
        std::string * temp = 0;

        temp = isa::utils::replace(&dmVarsTemplate, "<%PERIOD_NUM%>", period_s);
        temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
        temp = isa::utils::replace(temp, "<%DM_NUM%>", dm_s, true);
        dmVars_s->append(*temp);
        temp = isa::utils::replace(&computeTemplate, "<%PERIOD_NUM%>", period_s);
        temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
        temp = isa::utils::replace(temp, "<%DM_NUM%>", dm_s, true);
        compute_s->append(*temp);
        delete temp;
      }
    }
  }

  code = isa::utils::replace(code, "<%PERIOD_VARS%>", *periodVars_s, true);
  code = isa::utils::replace(code, "<%BIN_VARS%>", *binVars_s, true);
  code = isa::utils::replace(code, "<%DM_VARS%>", *dmVars_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);

  return code;
}

std::string * getFoldingPhi(const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, const unsigned int nrBinsPerThread) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "namespace PulsarSearch {\n"
    "template< typename T > void foldingPhi" + isa::utils::toString< unsigned int >(nrDMsPerThread) + "x" + isa::utils::toString< unsigned int >(nrPeriodsPerThread) + "x" + isa::utils::toString< unsigned int >(nrBinsPerThread) + "(const unsigned int second, const AstroData::Observation & observation, const float * const __restrict__ samples, float * const __restrict__ bins, const unsigned int * const __restrict__ readCounters, unsigned int * const __restrict__ writeCounters, const unsigned int * const __restrict__ samplesPerBin) {\n"
    "#pragma omp parallel for schedule(static)\n"
    "for ( unsigned int periodIndex = 0; periodIndex < observation.getNrPeriods(); periodIndex += " + isa::utils::toString< unsigned int >(nrPeriodsPerThread) + ") {\n"
    "<%PERIOD_VARS%>"
    "\n"
    "#pragma omp parallel for schedule(static)\n"
		"for ( unsigned int bin = 0; bin < observation.getNrBins(); bin += " + isa::utils::toString< unsigned int >(nrBinsPerThread) + ") {\n"
    "<%BIN_VARS%>"
    "\n"
    "#pragma omp parallel for schedule(static)\n"
    "for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm += 16 * " + isa::utils::toString< unsigned int >(nrDMsPerThread) + ") {\n"
    "<%DM_VARS%>"
    "\n"
    "<%COMPUTE%>"
    "}\n"
    "}\n"
    "}\n"
    "}\n"
    "}\n";
  std::string periodVarsTemplate = "const unsigned int periodValuep<%PERIOD_NUM%> = observation.getFirstPeriod() + ((periodIndex + <%PERIOD_NUM%>) * observation.getPeriodStep());\n";
  std::string binVarsTemplate = "const unsigned int pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> = readCounters[((periodIndex + <%PERIOD_NUM%>) * observation.getNrPaddedBins()) + (bin + <%BIN_NUM%>)];\n"
    "unsigned int foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>;\n"
    "unsigned int samplep<%PERIOD_NUM%>b<%BIN_NUM%>;\n";
  std::string dmVarsTemplate = "__m512 foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%> = _mm512_setzero_ps();\n";
  std::string computeTemplate = "foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%> = 0;\n"
    "samplep<%PERIOD_NUM%>b<%BIN_NUM%> = samplesPerBin->at(((periodIndex + <%PERIOD_NUM%>) * 2 * observation.getNrPaddedBins()) + ((bin + <%BIN_NUM%>) * 2) + 1) + ((pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> / samplesPerBin->at(((periodIndex + <%PERIOD_NUM%>) * 2 * observation.getNrPaddedBins()) + ((bin + <%BIN_NUM%>) * 2))) * periodValuep<%PERIOD_NUM%>) + (pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> % samplesPerBin->at(((periodIndex + <%PERIOD_NUM%>) * 2 * observation.getNrPaddedBins()) + ((bin + <%BIN_NUM%>) * 2)));\n"
    "\n"
    "if ( (samplep<%PERIOD_NUM%>b<%BIN_NUM%> / observation.getNrSamplesPerSecond()) == second ) {\n"
    "samplep<%PERIOD_NUM%>b<%BIN_NUM%> %= observation.getNrSamplesPerSecond();\n"
    "}\n"
    "while ( samplep<%PERIOD_NUM%>b<%BIN_NUM%> < observation.getNrSamplesPerSecond() ) {\n"
    "foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%> = _mm512_add_ps(foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%>, _mm512_load_ps(&(samples[(samplep<%PERIOD_NUM%>b<%BIN_NUM%> * observation.getNrPaddedDMs()) + (dm + <%DM_NUM%>)])));\n"
    "foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>++;\n"
    "\n"
    "if ( (foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%> + pCounterp<%PERIOD_NUM%>b<%BIN_NUM%>) % samplesPerBin->at(((periodIndex + <%PERIOD_NUM%>) * 2 * observation.getNrPaddedBins()) + ((bin + <%BIN_NUM%>) * 2)) == 0 ) {\n"
    "samplep<%PERIOD_NUM%>b<%BIN_NUM%> += periodValuep<%PERIOD_NUM%>;\n"
    "} else {\n"
    "samplep<%PERIOD_NUM%>b<%BIN_NUM%>++;\n"
    "}\n"
    "}\n"
    "\n"
    "if ( foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%> > 0 ) {\n"
    "const __m512 pValue = _mm512_load_ps(&(bins[((bin + <%BIN_NUM%>) * observation.getNrPeriods() * observation.getNrPaddedDMs()) + ((periodIndex + <%PERIOD_NUM%>) * observation.getNrPaddedDMs()) + (dm + <%DM_NUM%>)]));\n"
    "const __m512 cValue = _mm512_div_ps(_mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>), pValue), foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%>), _mm512_add_ps(_mm512_set1_ps(pCounterp<%PERIOD_NUM%>n<%BIN_NUM%>), _mm512_set1_ps(foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>)));\n"
    "_mm512_store_ps(&((bin + <%BIN_NUM%>)s[((bin + <%BIN_NUM%>) * observation.getNrPeriods() * observation.getNrPaddedDMs()) + ((periodIndex + <%PERIOD_NUM%>) * observation.getNrPaddedDMs()) + (dm + <%DM_NUM%>)]), cValue)\n"
    "writeCounters[((periodIndex + <%PERIOD_NUM%>) * observation.getNrPaddedBins()) + (bin + <%BIN_NUM%>)] = pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> + foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>;\n"
    "}\n";
  // End kernel's template
  
  std::string * periodVars_s = new std::string();
  std::string * binVars_s = new std::string();
  std::string * dmVars_s = new std::string();
  std::string * compute_s = new std::string();

  for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
    std::string period_s = isa::utils::toString< unsigned int >(period);
    std::string * temp = 0;

    temp = isa::utils::replace(&periodVarsTemplate, "<%PERIOD_NUM%>", period_s);
    periodVars_s->append(*temp);
    delete temp;

    for ( unsigned int bin = 0; bin < nrBinsPerThread; bin++ ) {
      std::string bin_s = isa::utils::toString< unsigned int >(bin);
      std::string * temp = 0;

      temp = isa::utils::replace(&binVarsTemplate, "<%PERIOD_NUM%>", period_s);
      temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
      binVars_s->append(*temp);
      delete temp;

      for ( unsigned int dm = 0; dm < nrDMsPerThread; dm++ ) {
        std::string dm_s = isa::utils::toString< unsigned int >(dm);
        std::string * temp = 0;

        temp = isa::utils::replace(&dmVarsTemplate, "<%PERIOD_NUM%>", period_s);
        temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
        temp = isa::utils::replace(temp, "<%DM_NUM%>", dm_s, true);
        dmVars_s->append(*temp);
        temp = isa::utils::replace(&computeTemplate, "<%PERIOD_NUM%>", period_s);
        temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
        temp = isa::utils::replace(temp, "<%DM_NUM%>", dm_s, true);
        compute_s->append(*temp);
        delete temp;
      }
    }
  }

  code = isa::utils::replace(code, "<%PERIOD_VARS%>", *periodVars_s, true);
  code = isa::utils::replace(code, "<%BIN_VARS%>", *binVars_s, true);
  code = isa::utils::replace(code, "<%DM_VARS%>", *dmVars_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);

  return code;
}

} // PulsarSearch

#endif // FOLDING_HPP
