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
std::string typeName("float");

void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< unsigned int > * samplesPerBin, cl::Buffer * samplesPerBin_d, std::vector< dataType > & dedispersedData, cl::Buffer * dedispersedData_d, std::vector< dataType > & foldedData, cl::Buffer * foldedData_d, std::vector< unsigned int > & readCounters, cl::Buffer * readCounters_d, std::vector< unsigned int > & writeCounters, cl::Buffer * writeCounters_d);

int main(int argc, char * argv[]) {
  bool reInit = true;
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
  unsigned int threadUnit = 0;
  unsigned int threadIncrement = 0;
	unsigned int minThreads = 0;
	unsigned int maxThreadsPerBlock = 0;
	unsigned int maxItemsPerThread = 0;
	unsigned int maxColumns = 0;
	unsigned int maxRows = 0;
  unsigned int maxVector = 0;
  PulsarSearch::FoldingConf conf;
  AstroData::Observation observation;
  cl::Event event;

	try {
    isa::utils::ArgumentList args(argc, argv);

		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    threadUnit = args.getSwitchArgument< unsigned int >("-thread_unit");
    threadIncrement = args.getSwitchArgument< unsigned int >("-thread_inc");
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxThreadsPerBlock = args.getSwitchArgument< unsigned int >("-max_threads");
		maxItemsPerThread = args.getSwitchArgument< unsigned int >("-max_items");
		maxColumns = args.getSwitchArgument< unsigned int >("-max_columns");
		maxRows = args.getSwitchArgument< unsigned int >("-max_rows");
    maxVector = args.getSwitchArgument< unsigned int >("-max_vector");
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
    observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
    observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), args.getSwitchArgument< unsigned int >("-first_period"), args.getSwitchArgument< unsigned int >("-period_step"));
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
	} catch ( isa::utils::EmptyCommandLine &err ) {
		std::cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -thread_unit ... -thread_inc ... -padding ... -min_threads ... -max_threads ... -max_items ... -max_columns ... -max_rows ... -max_vector ... -samples ... -dms ... -periods ... -bins ... -first_period ... -period_step ..." << std::endl;
		return 1;
	} catch ( std::exception &err ) {
		std::cerr << err.what() << std::endl;
		return 1;
	}

  // Allocate memory
  cl::Buffer samplesPerBin_d;
  cl::Buffer dedispersedData_d;
  cl::Buffer foldedData_d;
  cl::Buffer readCounters_d;
  cl::Buffer writeCounters_d;
  std::vector< unsigned int > * samplesPerBin = PulsarSearch::getSamplesPerBin(observation);
  std::vector< dataType > dedispersedData = std::vector< dataType >(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
  std::vector< dataType > foldedData = std::vector< dataType >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  std::vector< unsigned int > readCounters = std::vector< unsigned int >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  std::vector< unsigned int > writeCounters = std::vector< unsigned int >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());

	srand(time(0));
  for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
    for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
      dedispersedData[(sample * observation.getNrPaddedDMs()) + DM] = static_cast< dataType >(rand() % 10);
		}
	}
  std::fill(foldedData.begin(), foldedData.end(), static_cast< dataType >(0));
  std::fill(readCounters.begin(), readCounters.end(), 0);
  std::fill(writeCounters.begin(), writeCounters.end(), 0);

	// Initialize OpenCL
	cl::Context clContext;
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);

	// Find the parameters
	std::vector< unsigned int > DMsPerBlock;
	for ( unsigned int DMs = minThreads; DMs <= maxColumns; DMs += threadIncrement ) {
		if ( (observation.getNrDMs() % DMs) == 0 || (observation.getNrPaddedDMs() % DMs) == 0 ) {
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
	std::cout << "# nrDMs nrSamples nrPeriods nrBins firstPeriod periodStep DMsPerBlock periodsPerBlock binsPerBlock DMsPerThread periodsPerThread binsPerThread vector GFLOP/s time stdDeviation COV" << std::endl << std::endl;

  for ( std::vector< unsigned int >::iterator DMs = DMsPerBlock.begin(); DMs != DMsPerBlock.end(); ++DMs ) {
    conf.setNrDMsPerBlock(*DMs);
    for ( std::vector< unsigned int >::iterator periods = periodsPerBlock.begin(); periods != periodsPerBlock.end(); ++periods ) {
      conf.setNrPeriodsPerBlock(*periods);
      for ( std::vector< unsigned int >::iterator bins = binsPerBlock.begin(); bins != binsPerBlock.end(); ++bins ) {
        conf.setNrBinsPerBlock(*bins);
        if ( conf.getNrDMsPerBlock() * conf.getNrPeriodsPerBlock() * conf.getNrBinsPerBlock() > maxThreadsPerBlock ) {
          break;
        } else if ( conf.getNrPeriodsPerBlock() * conf.getNrBinsPerBlock() > maxRows ) {
          break;
        } else if ( (conf.getNrDMsPerBlock() * conf.getNrPeriodsPerBlock() * conf.getNrBinsPerBlock()) % threadUnit != 0 ) {
          break;
        }
        for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
          conf.setNrDMsPerThread(DMsPerThread);
          if ( observation.getNrDMs() % (conf.getNrDMsPerBlock() * conf.getNrDMsPerThread()) != 0 && observation.getNrPaddedDMs() % (conf.getNrDMsPerBlock() * conf.getNrDMsPerThread()) != 0) {
            continue;
          }
          for ( unsigned int periodsPerThread = 1; periodsPerThread <= maxItemsPerThread; periodsPerThread++ ) {
            conf.setNrPeriodsPerThread(periodsPerThread);
            if ( observation.getNrPeriods() % (conf.getNrPeriodsPerBlock() * conf.getNrPeriodsPerThread()) != 0 ) {
              continue;
            }
            for ( unsigned int binsPerThread = 1; binsPerThread <= maxItemsPerThread; binsPerThread++ ) {
              conf.setNrBinsPerThread(binsPerThread);
              if ( observation.getNrBins() % (conf.getNrBinsPerBlock() * conf.getNrBinsPerThread()) != 0 ) {
                continue;
              }
              for ( unsigned int vector = 1; vector <= maxVector; vector *= 2 ) {
                conf.setVector(vector);
                if ( observation.getNrDMs() % (conf.getNrDMsPerBlock() * conf.getNrDMsPerThread() * conf.getVector()) != 0 && observation.getNrPaddedDMs() % (conf.getNrDMsPerBlock() * conf.getNrDMsPerThread() * conf.getVector()) != 0) {
                  continue;
                }
                if ( 1 + (2 * periodsPerThread) + binsPerThread + DMsPerThread + (4 * periodsPerThread * binsPerThread) + (vector * periodsPerThread * binsPerThread * DMsPerThread) > maxItemsPerThread ) {
                  break;
                }

                // Generate kernel
                double flops = isa::utils::giga(static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrSamplesPerSecond());
                isa::utils::Timer timer;
                cl::Kernel * kernel;
                std::string * code = PulsarSearch::getFoldingOpenCL(conf, typeName, observation);
                
                if ( reInit ) {
                  delete clQueues;
                  clQueues = new std::vector< std::vector< cl::CommandQueue > >();
                  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
                  try {
                    initializeDeviceMemory(clContext, &(clQueues->at(clDeviceID)[0]), samplesPerBin, &samplesPerBin_d, dedispersedData, &dedispersedData_d, foldedData, &foldedData_d, readCounters, &readCounters_d, writeCounters, &writeCounters_d);
                  } catch ( cl::Error & err ) {
                    return -1;
                  }
                  reInit = false;
                }
                try {
                  kernel = isa::OpenCL::compile("folding", *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
                } catch ( isa::OpenCL::OpenCLError & err ) {
                  std::cerr << err.what() << std::endl;
                  delete code;
                  break;
                }
                delete code;

                unsigned int nrThreads = 0;
                if ( observation.getNrDMs() % (conf.getNrDMsPerBlock() * conf.getNrDMsPerThread() * conf.getVector()) == 0 ) {
                  nrThreads = observation.getNrDMs() / conf.getNrDMsPerThread() / conf.getVector();
                } else {
                  nrThreads = observation.getNrPaddedDMs() / conf.getNrDMsPerThread() / conf.getVector();
                }
                cl::NDRange global(nrThreads, observation.getNrPeriods() / conf.getNrPeriodsPerThread(), observation.getNrBins() / conf.getNrBinsPerThread());
                cl::NDRange local(conf.getNrDMsPerBlock(), conf.getNrPeriodsPerBlock(), conf.getNrBinsPerBlock());

                kernel->setArg(0, 0);
                kernel->setArg(1, dedispersedData_d);
                kernel->setArg(2, foldedData_d);
                kernel->setArg(3, readCounters_d);
                kernel->setArg(4, writeCounters_d);
                kernel->setArg(5, samplesPerBin_d);

                try {
                  // Warm-up run
                  clQueues->at(clDeviceID)[0].finish();
                  clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
                  event.wait();
                  // Tuning runs
                  for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
                    timer.start();
                    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
                    event.wait();
                    timer.stop();
                  }
                } catch ( cl::Error & err ) {
                  std::cerr << "OpenCL error kernel execution (";
                  std::cerr << conf.print() << "): ";
                  std::cerr << isa::utils::toString(err.err()) << "." << std::endl;
                  delete kernel;
                  if ( err.err() == -4 || err.err() == -61 ) {
                    return -1;
                  }
                  reInit = true;
                  break;
                }
                delete kernel;

                std::cout << observation.getNrDMs() << " " << observation.getNrSamplesPerSecond() << " " << observation.getNrPeriods() << " " << observation.getNrBins() << " " << observation.getFirstPeriod() << " " << observation.getPeriodStep() << " ";
                std::cout << conf.print() << " ";
                std::cout << std::setprecision(3);
                std::cout << flops / timer.getAverageTime() << " ";
                std::cout << std::setprecision(6);
                std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation() << std::endl;
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

void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< unsigned int > * samplesPerBin, cl::Buffer * samplesPerBin_d, std::vector< dataType > & dedispersedData, cl::Buffer * dedispersedData_d, std::vector< dataType > & foldedData, cl::Buffer * foldedData_d, std::vector< unsigned int > & readCounters, cl::Buffer * readCounters_d, std::vector< unsigned int > & writeCounters, cl::Buffer * writeCounters_d) {
  try {
    *samplesPerBin_d = cl::Buffer(clContext, CL_MEM_READ_ONLY, samplesPerBin->size() * sizeof(unsigned int), 0, 0);
    *dedispersedData_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, dedispersedData.size() * sizeof(dataType), 0, 0);
    *foldedData_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, foldedData.size() * sizeof(dataType), 0, 0);
    *readCounters_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, readCounters.size() * sizeof(unsigned int), 0, 0);
    *writeCounters_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, writeCounters.size() * sizeof(unsigned int), 0, 0);
    clQueue->enqueueWriteBuffer(*samplesPerBin_d, CL_FALSE, 0, samplesPerBin->size() * sizeof(unsigned int), reinterpret_cast< void * >(samplesPerBin->data()));
    clQueue->enqueueWriteBuffer(*dedispersedData_d, CL_FALSE, 0, dedispersedData.size() * sizeof(dataType), reinterpret_cast< void * >(dedispersedData.data()));
    clQueue->enqueueWriteBuffer(*foldedData_d, CL_FALSE, 0, foldedData.size() * sizeof(dataType), reinterpret_cast< void * >(foldedData.data()));
    clQueue->enqueueWriteBuffer(*readCounters_d, CL_FALSE, 0, readCounters.size() * sizeof(unsigned int), reinterpret_cast< void * >(readCounters.data()));
    clQueue->enqueueWriteBuffer(*writeCounters_d, CL_FALSE, 0, writeCounters.size() * sizeof(unsigned int), reinterpret_cast< void * >(writeCounters.data()));
    clQueue->finish();
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString(err.err()) << "." << std::endl;
    throw;
  }
}

