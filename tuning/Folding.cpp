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
#include <algorithm>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <Bins.hpp>
#include <Folding.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>

typedef float dataType;
string typeName("float");


int main(int argc, char * argv[]) {
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int minThreads = 0;
	unsigned int maxThreadsPerBlock = 0;
	unsigned int maxItemsPerThread = 0;
	unsigned int maxColumns = 0;
	unsigned int maxRows = 0;
  unsigned int maxVector = 0;
  AstroData::Observation< dataType > observation("FoldingTuning", typeName);

	try {
    isa::utils::ArgumentList args(argc, argv);

		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxThreadsPerBlock = args.getSwitchArgument< unsigned int >("-max_threads");
		maxItemsPerThread = args.getSwitchArgument< unsigned int >("-max_items");
		maxColumns = args.getSwitchArgument< unsigned int >("-max_columns");
		maxRows = args.getSwitchArgument< unsigned int >("-max_rows");
    maxVector = args.getSwitchArgument< unsigned int >("-max_vector");
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dms"));
    observation.setNrPeriods(args.getSwitchArgument< unsigned int >("-periods"));
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
    observation.setFirstPeriod(args.getSwitchArgument< unsigned int >("-first_period"));
    observation.setPeriodStep(args.getSwitchArgument< unsigned int >("-period_step"));
	} catch ( isa::utils::EmptyCommandLine &err ) {
		std::cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -padding ... -min_threads ... -max_threads ... -max_items ... -max_columns ... -max_rows ... -max_vector ... -samples ... -dms ... -periods ... -bins ... -first_period ... -period_step ..." << std::endl;
		return 1;
	} catch ( std::exception &err ) {
		std::cerr << err.what() << std::endl;
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
  std::vector< dataType > dedispersedData = std::vector< dataType >(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
  cl::Buffer dedispersedData_d;
  std::vector< dataType > foldedData = std::vector< dataType >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  cl::Buffer foldedData_d;
  std::vector< unsigned int > readCounters = std::vector< unsigned int >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  cl::Buffer readCounters_d;
  std::vector< unsigned int > writeCounters = std::vector< unsigned int >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  cl::Buffer writeCounters_d;
  try {
    samplesPerBin_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, samplesPerBin->size() * sizeof(unsigned int), NULL, NULL);
    dedispersedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, dedispersedData.size() * sizeof(dataType), NULL, NULL);
    foldedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, foldedData.size() * sizeof(dataType), NULL, NULL);
    readCounters_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, readCounters.size() * sizeof(unsigned int), NULL, NULL);
    writeCounters_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, writeCounters.size() * sizeof(unsigned int), NULL, NULL);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(NULL));
  for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
    for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
      dedispersedData[(sample * observation.getNrPaddedDMs()) + DM] = static_cast< dataType >(rand() % 10);
		}
	}
  std::fill(foldedData.begin(), foldedData.end(), static_cast< dataType >(0));
  std::fill(readCounters.begin(), readCounters.end(), 0);
  std::fill(writeCounters.begin(), writeCounters.end(), 0);

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(samplesPerBin_d, CL_FALSE, 0, samplesPerBin->size() * sizeof(unsigned int), reinterpret_cast< void * >(samplesPerBin->data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dedispersedData_d, CL_FALSE, 0, dedispersedData.size() * sizeof(dataType), reinterpret_cast< void * >(dedispersedData.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(foldedData_d, CL_FALSE, 0, foldedData.size() * sizeof(dataType), reinterpret_cast< void * >(foldedData.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(readCounters_d, CL_FALSE, 0, readCounters.size() * sizeof(unsigned int), reinterpret_cast< void * >(readCounters.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(writeCounters_d, CL_FALSE, 0, writeCounters.size() * sizeof(unsigned int), reinterpret_cast< void * >(writeCounters.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	// Find the parameters
	std::vector< unsigned int > DMsPerBlock;
	for ( unsigned int DMs = minThreads; DMs <= maxColumns; DMs += minThreads ) {
		if ( (observation.getNrPaddedDMs() % DMs) == 0 ) {
			DMsPerBlock.push_back(DMs);
		}
	}
	std::vector< unsigned int > periodsPerBlock;
	for ( unsigned int periods = 1; periods <= maxRows; periods++ ) {
		if ( (observation.getNrPeriods() % periods) == 0 ) {
			periodsPerBlock.push_back(periods);
		}
	}
	std::vector< unsigned int > binsPerBlock;
	for ( unsigned int bins = 1; bins <= maxRows; bins++ ) {
		if ( (observation.getNrBins() % bins) == 0 ) {
			binsPerBlock.push_back(bins);
		}
	}

	std::cout << std::fixed << std::endl;
	std::cout << "# nrDMs nrSamples nrPeriods nrBins firstPeriod periodStep DMsPerBlock periodsPerBlock binsPerBlock DMsPerThread periodsPerThread binsPerThread vector GFLOP/s err time err" << std::endl << std::endl;

  for ( std::vector< unsigned int >::iterator DMs = DMsPerBlock.begin(); DMs != DMsPerBlock.end(); ++DMs ) {
    for ( std::vector< unsigned int >::iterator periods = periodsPerBlock.begin(); periods != periodsPerBlock.end(); ++periods ) {
      for ( std::vector< unsigned int >::iterator bins = binsPerBlock.begin(); bins != binsPerBlock.end(); ++bins ) {
        if ( (*DMs * *periods * *bins) > maxThreadsPerBlock ) {
          break;
        }
        for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
          if ( observation.getNrPaddedDMs() % (*DMs * DMsPerThread) != 0 ) {
            continue;
          }
          for ( unsigned int periodsPerThread = 1; periodsPerThread <= maxItemsPerThread; periodsPerThread++ ) {
            if ( observation.getNrPeriods() % (*periods * periodsPerThread) != 0 ) {
              continue;
            }
            for ( unsigned int binsPerThread = 1; binsPerThread <= maxItemsPerThread; binsPerThread++ ) {
              if ( observation.getNrBins() % (*bins * binsPerThread) != 0 ) {
                continue;
              }
              if ( 1 + (2 * periodsPerThread) + binsPerThread + DMsPerThread + (4 * periodsPerThread * binsPerThread) + (vector * periodsPerThread * binsPerThread * DMsPerThread) > maxItemsPerThread ) {
                break;
              }
              for ( unsigned int vector = 1; vector < maxVector; vector *= 2 ) {
                if ( observation.getNrPaddedDMs() % (*DMs * DMsPerThread * vector) != 0 ) {
                  continue;
                }
                // Generate kernel
                cl::Kernel * kernel;
                std::string * code = PulsarSearch::getFoldingOpenCL(*DMs, *periods, *bins, DMsPerThread, periodsPerThread, binsPerThread, vector, typeName, observation);

                try {
                  kernel = isa::OpenCL::compile("folding", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
                } catch ( isa::OpenCL::OpenCLError &err ) {
                  std::cerr << err.what() << std::endl;
                  continue;
                }

                cl::NDRange global(observation.getNrPaddedDMs() / vector / DMsPerThread, observation.getNrPeriods() / periodsPerThread, observation.getNrBins() / binsPerThread);
                cl::NDRange local(*DMs, *periods, *bins);

                kernel->setArg(0, 0);
                kernel->setArg(1, dedispersedData_d);
                kernel->setArg(2, foldedData_d);
                kernel->setArg(3, readCounters_d);
                kernel->setArg(4, writeCounters_d);
                kernel->setArg(5, samplesPerBin_d);

                // Warm-up run
                try {
                  clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
                } catch ( cl::Error &err ) {
                  std::cerr << "OpenCL error kernel execution: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
                  continue;
                }
                // Tuning runs
                double flops = isa::utils::giga(static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrSamplesPerSecond());
                isa::utils::Timer timer("Kernel Timer");
                isa::utils::Stats< double > stats;
                cl::Event event;

                try {
                  for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
                    timer.start();
                    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, NULL, &event);
                    event.wait();
                    timer.stop();
                    stats.addElement(flops / timer.getLastRunTime());
                  }
                } catch ( cl::Error &err ) {
                  std::cerr << "OpenCL error kernel execution: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
                  continue;
                }

                std::cout << observation.getNrDMs() << " " << observation.getNrSamplesPerSecond() << " " << observation.getNrPeriods() << " " << observation.getNrBins() << " " << observation.getFirstPeriod() << " " << observation.getPeriodStep() << " " << *DMs << " " << *periods << " " << *bins << " " << DMsPerThread << " " << periodsPerThread << " " << binsPerThread << " " << vector << " " << std::setprecision(3) << stats.getAverage() << " " << stats.getStdDev() << " " << std::setprecision(6) << timer.getAverageTime() << " " << timer.getStdDev() << std::endl;
              }
            }
          }
        }
      }
    }
  }

	std::cout << std::endl;

	return 0;
}

