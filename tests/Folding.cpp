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
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ctime>

#include <ArgumentList.hpp>
#include <Exceptions.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <Bins.hpp>
#include <Folding.hpp>

typedef float dataType;
std::string typeName("float");


int main(int argc, char *argv[]) {
  bool print = false;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
  unsigned int nrDMsPerBlock = 0;
  unsigned int nrPeriodsPerBlock = 0;
  unsigned int nrBinsPerBlock = 0;
  unsigned int nrDMsPerThread = 0;
  unsigned int nrPeriodsPerThread = 0;
  unsigned int nrBinsPerThread = 0;
	long long unsigned int wrongSamples = 0;
	AstroData::Observation< dataType > observation("FoldingTest", typeName);

	try {
    isa::utils::ArgumentList args(argc, argv);
    print = args.getSwitch("-print");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    nrDMsPerBlock = args.getSwitchArgument< unsigned int >("-db");
    nrPeriodsPerBlock = args.getSwitchArgument< unsigned int >("-pb");
    nrBinsPerBlock = args.getSwitchArgument< unsigned int >("-bb");
    nrDMsPerThread = args.getSwitchArgument< unsigned int >("-dt");
    nrPeriodsPerThread = args.getSwitchArgument< unsigned int >("-pt");
    nrBinsPerThread = args.getSwitchArgument< unsigned int >("-bt");
    observation.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dms"));
    observation.setNrPeriods(args.getSwitchArgument< unsigned int >("-periods"));
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
    observation.setFirstPeriod(args.getSwitchArgument< unsigned int >("-first_period"));
    observation.setPeriodStep(args.getSwitchArgument< unsigned int >("-period_step"));
	} catch  ( isa::Exceptions::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-print] -opencl_platform ... -opencl_device ... -padding ... -db ... -pb ... -bb ... -dt ... -pt ... -bt ... -seconds .... -samples ... -dms ... -periods ... -bins ... -first_period ... -period_step ..." << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);
  std::vector< unsigned int > * samplesPerBin = PulsarSearch::getSamplesPerBin(observation);

	// Allocate memory
  cl::Buffer samplesPerBin_d;
  std::vector< std::vector< dataType > * > dedispersedData = std::vector< std::vector< dataType > * >(observation.getNrSeconds());
  std::vector< std::vector< dataType > * > dedispersedData_c = std::vector< std::vector< dataType > * >(observation.getNrSeconds());
  for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
    dedispersedData.at(second) = new std::vector< dataType >(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
    dedispersedData_c.at(second) = new std::vector< dataType >(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond());
  }
  cl::Buffer dedispersedData_d;
  std::vector< dataType > foldedData_c = std::vector< dataType >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  std::vector< dataType > foldedData = std::vector< dataType >(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
  cl::Buffer foldedData_d;
  std::vector< unsigned int > counters_c = std::vector< unsigned int >(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
  std::vector< unsigned int > readCounters = std::vector< unsigned int >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  cl::Buffer readCounters_d;
  std::vector< unsigned int > writeCounters = std::vector< unsigned int >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  cl::Buffer writeCounters_d;
  try {
    samplesPerBin_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, samplesPerBin->size() * sizeof(unsigned int), NULL, NULL);
    dedispersedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, (dedispersedData.at(0))->size() * sizeof(dataType), NULL, NULL);
    foldedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, foldedData.size() * sizeof(dataType), NULL, NULL);
    readCounters_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, readCounters.size() * sizeof(unsigned int), NULL, NULL);
    writeCounters_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, writeCounters.size() * sizeof(unsigned int), NULL, NULL);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(NULL));
  for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
        (dedispersedData.at(second))->a((sample * observation.getNrPaddedDMs()) + dm) = static_cast< dataType >(rand() % 10);
        dedispersedData_c.at(second)[(dm * observation.getNrSamplesPerPaddedSecond()) + sample] = dedispersedData.at(second)[(sample * observation.getNrPaddedDMs()) + dm];
      }
    }
  }
  std::fill(foldedData.begin(), foldedData.end(), static_cast< dataType >(0));
  std::fill(foldedData_c.begin(), foldedData_c.end(), static_cast< dataType >(0));
  std::fill(counters_c.begin(), counters_c.end(), 0);
  std::fill(readCounters.begin(), readCounters.end(), 0);
  std::fill(writeCounters.begin(), writeCounters.end(), 0);

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(samplesPerBin_d, CL_FALSE, 0, samplesPerBin->size() * sizeof(unsigned int), reinterpret_cast< void * >(samplesPerBin->data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(foldedData_d, CL_FALSE, 0, foldedData.size() * sizeof(dataType), reinterpret_cast< void * >(foldedData.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(readCounters_d, CL_FALSE, 0, readCounters->size() * sizeof(unsigned int), reinterpret_cast< void * >(readCounters.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(writeCounters_d, CL_FALSE, 0, writeCounters->size() * sizeof(unsigned int), reinterpret_cast< void * >(writeCounters.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  cl::Kernel * kernel;
  std::string * code = PulsarSearch::getFoldingOpenCL(nrDMsPerBlock, nrPeriodsPerBlock, nrBinsPerBlock, nrDMsPerThread, nrPeriodsPerThread, nrBinsPerThread, typeName, observation);
  if ( print ) {
    std::cout << *code << std::endl;
  }

  try {
    kernel = isa::OpenCL::compile("folding", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
  } catch ( isa::Exceptions::OpenCLError &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  try {
    cl::NDRange global(observation.getNrPaddedDMs() / nrDMsPerThread, observation.getNrPeriods() / nrPeriodsPerThread, observation.getNrBins() / nrBinsPerThread);
    cl::NDRange local(nrDMsPerBlock, nrPeriodsPerBlock, nrBinsPerBlock);

    kernel->setArg(1, dedispersedData_d);
    kernel->setArg(2, foldedData_d);
    kernel->setArg(5, samplesPerBin_d);
    for ( unsigned int second = 0; second < observation.getNrSeconds(); second++ ) {
      kernel->setArg(0, second);
      if ( second % 2 == 0 ) {
        kernel->setArg(3, readCounters_d);
        kernel->setArg(4, writeCounters_d);
      } else {
        kernel->setArg(3, writeCounters_d);
        kernel->setArg(4, readCounters_d);
      }
      clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dedispersedData_d, CL_FALSE, 0, (dedispersedData.at(second))->size() * sizeof(dataType), reinterpret_cast< void * >((dedispersedData.at(second))->data()));
      clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
      PulsarSearch::folding(second, observation, *(dedispersedData_c.at(second)), foldedData_c, counters_c);
    }
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(foldedData_d, CL_TRUE, 0, foldedData.size() * sizeof(dataType), reinterpret_cast< void * >(foldedData.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
      for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
        if ( ! isa::utils::same(foldedData_c[(dm * observation.getNrPeriods() * observation.getNrPaddedBins()) + (period * observation.getNrPaddedBins()) + bin], foldedData[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) * (period * observation.getNrPaddedDMs()) + dm]) ) {
          wrongSamples++;
        }
      }
    }
  }

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins()) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

