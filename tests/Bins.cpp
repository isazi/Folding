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

#include <iostream>
#include <vector>

#include <Bins.hpp>
#include <Observation.hpp>
#include <utils.hpp>

const unsigned int padding = 32;
const unsigned int nrPeriods = 10;
const unsigned int firstPeriod = 40;
const unsigned int periodStep = 40;
const unsigned int nrBins = 128;


int main(int argc, char *argv[]) {
  std::vector< unsigned int > * samplesPerBin = 0;
  AstroData::Observation< float > obs("BinsTest", "float");

  obs.setPadding(padding);
  obs.setNrPeriods(nrPeriods);
  obs.setFirstPeriod(firstPeriod);
  obs.setPeriodStep(periodStep);
  obs.setNrBins(nrBins);
  samplesPerBin = PulsarSearch::getNrSamplesPerBin(obs);
  
  std::cout << "Samples/bin:" << std::endl;
  for ( unsigned int period = 0; period < obs.getNrPeriods(); period++ ) {
    for ( unsigned int bin = 0; bin < obs.getNrBins(); bin++ ) {
      std::cout << samplesPerBin->at((period * obs.getNrBins() * isa::utils::pad(2, padding)) + (bin * isa::utils::pad(2, padding))) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Offsets:" << std::endl;
  for ( unsigned int period = 0; period < obs.getNrPeriods(); period++ ) {
    for ( unsigned int bin = 0; bin < obs.getNrBins(); bin++ ) {
      std::cout << samplesPerBin->at((period * obs.getNrBins() * isa::utils::pad(2, padding)) + (bin * isa::utils::pad(2, padding)) + 1) << "\t";
    }
    std::cout << std::endl;
  }

  return 0;
}

