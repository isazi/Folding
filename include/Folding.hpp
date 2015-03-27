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
#include <fstream>

#include <utils.hpp>
#include <Observation.hpp>


#ifndef FOLDING_HPP
#define FOLDING_HPP

namespace PulsarSearch {

class FoldingConf {
public:
  FoldingConf();
  ~FoldingConf();

  // Get
  unsigned int getNrDMsPerBlock() const;
  unsigned int getNrDMsPerThread() const;
  unsigned int getNrPeriodsPerBlock() const;
  unsigned int getNrPeriodsPerThread() const;
  unsigned int getNrBinsPerBlock() const;
  unsigned int getNrBinsPerThread() const;
  unsigned int getVector() const;
  // Set
  void setNrDMsPerBlock(unsigned int dms);
  void setNrDMsPerThread(unsigned int dms);
  void setNrPeriodsPerBlock(unsigned int periods);
  void setNrPeriodsPerThread(unsigned int periods);
  void setNrBinsPerBlock(unsigned int bins);
  void setNrBinsPerThread(unsigned int bins);
  void setVector(unsigned int vector);
  // utils
  std::string print() const;

private:
  unsigned int nrDMsPerBlock;
  unsigned int nrDMsPerThread;
  unsigned int nrPeriodsPerBlock;
  unsigned int nrPeriodsPerThread;
  unsigned int nrBinsPerBlock;
  unsigned int nrBinsPerThread;
  unsigned int vector;
};

typedef std::map< std::string, std::map< unsigned int, std::map< unsigned int, PulsarSearch::FoldingConf > > > tunedFoldingConf;

// Sequential folding
template< typename T > void folding(const unsigned int second, const AstroData::Observation & observation, const std::vector< T > & samples, std::vector< T > & bins, std::vector< unsigned int > & counters);
// OpenCL folding algorithm
std::string * getFoldingOpenCL(const FoldingConf & conf, const std::string & dataType, const AstroData::Observation & observation);
// Read configuration files
void readTunedFoldingConf(tunedFoldingConf & tunedFolding, const std::string & foldingFilename);


// Implementations
inline unsigned int FoldingConf::getNrDMsPerBlock() const {
  return nrDMsPerBlock;
}

inline unsigned int FoldingConf::getNrDMsPerThread() const {
  return nrDMsPerThread;
}

inline unsigned int FoldingConf::getNrPeriodsPerBlock() const {
  return nrPeriodsPerBlock;
}

inline unsigned int FoldingConf::getNrPeriodsPerThread() const {
  return nrPeriodsPerThread;
}

inline unsigned int FoldingConf::getNrBinsPerBlock() const {
  return nrBinsPerBlock;
}

inline unsigned int FoldingConf::getNrBinsPerThread() const {
  return nrBinsPerThread;
}

inline unsigned int FoldingConf::getVector() const {
  return vector;
}

inline void FoldingConf::setNrDMsPerBlock(unsigned int dms) {
  nrDMsPerBlock = dms;
}

inline void FoldingConf::setNrDMsPerThread(unsigned int dms) {
  nrDMsPerThread = dms;
}

inline void FoldingConf::setNrPeriodsPerBlock(unsigned int periods) {
  nrPeriodsPerBlock = periods;
}

inline void FoldingConf::setNrPeriodsPerThread(unsigned int periods) {
  nrPeriodsPerThread = periods;
}

inline void FoldingConf::setNrBinsPerBlock(unsigned int bins) {
  nrBinsPerBlock = bins;
}

inline void FoldingConf::setNrBinsPerThread(unsigned int bins) {
  nrBinsPerThread = bins;
}

inline void FoldingConf::setVector(unsigned int vector) {
  this->vector = vector;
}

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

} // PulsarSearch

#endif // FOLDING_HPP
