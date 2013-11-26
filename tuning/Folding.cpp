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

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::exception;
using std::ofstream;
using std::fixed;
using std::setprecision;
using std::numeric_limits;

#include <ArgumentList.hpp>
using isa::utils::ArgumentList;
#include <Observation.hpp>
using AstroData::Observation;
#include <InitializeOpenCL.hpp>
using isa::OpenCL::initializeOpenCL;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <utils.hpp>
using isa::utils::toStringValue;
#include <Folding.hpp>
using PulsarSearch::Folding;
#include <Timer.hpp>
using isa::utils::Timer;
#include <Bins.hpp>
using PulsarSearch::getNrSamplesPerBin;

typedef float dataType;
const string typeName("float");
const unsigned int maxThreadsPerBlock = 1024;
const unsigned int maxThreadsMultiplier = 512;
const unsigned int maxItemsPerThread = 16;
const unsigned int maxItemsMultiplier = 16;
const unsigned int padding = 32;


int main(int argc, char * argv[]) {
	unsigned int lowerNrThreads = 0;
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	Observation< dataType > observation("FoldingTuning", typeName);
	CLData< dataType > * dedispersedData = new CLData< dataType >("DedispersedData", true);
	CLData< dataType > * foldedData = new CLData<dataType >("FoldedData", true);
	CLData< unsigned int > * readCounterData = new CLData< unsigned int >("ReadCounterData", true);
	CLData< unsigned int > * writeCounterData = new CLData< unsigned int >("WriteCounterData", true);
	CLData< unsigned int > * nrSamplesPerBin = new CLData< unsigned int >("SamplesPerBin", true);


	try {
		ArgumentList args(argc, argv);

		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dms"));
		lowerNrThreads = args.getSwitchArgument< unsigned int >("-lnt");
		observation.setNrPeriods(args.getSwitchArgument< unsigned int >("-periods"));
		observation.setFirstPeriod(args.getSwitchArgument< unsigned int >("-first_period"));
		observation.setPeriodStep(args.getSwitchArgument< unsigned int >("-period_step"));
		observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
		observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
	} catch ( exception & err ) {
		cerr << err.what() << endl;
		return 1;
	}

	// Setup of the observation
	observation.setPadding(padding);
	
	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();
	
	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	cout << fixed << endl;
	cout << "# nrDMs nrPeriods firstPeriod periodStep nrBins nrSamplesPerSecond nrDMsPerBlock nrPeriodsPerBlock nrBinsPerBlock nrDMsPerThread nrPeriodsPerThread nrBinsPerThread GFLOP/s err time err" << endl << endl;

	// Allocate memory
	dedispersedData->allocateHostData(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
	dedispersedData->blankHostData();
	foldedData->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
	foldedData->blankHostData();
	readCounterData->allocateHostData(observation.getNrPeriods() * 2 * observation.getNrPaddedBins());
	readCounterData->blankHostData();
	writeCounterData->allocateHostData(observation.getNrPeriods() * 2 * observation.getNrPaddedBins());
	writeCounterData->blankHostData();
	vector< unsigned int > * nrSamplesPerBinData = getNrSamplesPerBin(observation);
	nrSamplesPerBin->allocateHostData(*nrSamplesPerBinData);

	dedispersedData->setCLContext(clContext);
	dedispersedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	foldedData->setCLContext(clContext);
	foldedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	readCounterData->setCLContext(clContext);
	readCounterData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	writeCounterData->setCLContext(clContext);
	writeCounterData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	nrSamplesPerBin->setCLContext(clContext);
	nrSamplesPerBin->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	nrSamplesPerBin->setDeviceReadOnly();

	try {
		dedispersedData->allocateDeviceData();
		dedispersedData->copyHostToDevice();
		foldedData->allocateDeviceData();
		foldedData->copyHostToDevice();
		readCounterData->allocateDeviceData();
		readCounterData->copyHostToDevice();
		writeCounterData->allocateDeviceData();
		writeCounterData->copyHostToDevice();
		nrSamplesPerBin->allocateDeviceData();
		nrSamplesPerBin->copyHostToDevice();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
	}

	// Find the parameters
	vector< vector< unsigned int > > configurations;
	for ( unsigned int DMsPerBlock = lowerNrThreads; DMsPerBlock <= maxThreadsPerBlock; DMsPerBlock++ ) {
		if ( observation.getNrDMs() % DMsPerBlock != 0 ) {
			continue;
		}

		for ( unsigned int periodsPerBlock = 1; periodsPerBlock <= maxThreadsMultiplier; periodsPerBlock++ ) {
			if ( observation.getNrPeriods() % periodsPerBlock != 0 ) {
				continue;
			} else if ( DMsPerBlock * periodsPerBlock > maxThreadsPerBlock ) {
				break;
			}

			for ( unsigned int binsPerBlock = 1; binsPerBlock <= maxThreadsMultiplier; binsPerBlock++ ) {
				if ( observation.getNrBins() % binsPerBlock != 0 ) {
					continue;
				} else if ( DMsPerBlock * periodsPerBlock * binsPerBlock > maxThreadsPerBlock ) {
					break;
				}

				for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
					if ( observation.getNrDMs() % (DMsPerBlock * DMsPerThread) != 0 ) {
						continue;
					}

					for ( unsigned int periodsPerThread = 1; periodsPerThread <= maxItemsMultiplier; periodsPerThread++ ) {
						if ( observation.getNrPeriods() % (periodsPerBlock * periodsPerThread) != 0 ) {
							continue;
						} else if ( (DMsPerThread + (2 * periodsPerThread)) > maxItemsPerThread ) {
							break;
						}

						for ( unsigned int binsPerThread = 1; binsPerThread <= maxItemsMultiplier; binsPerThread++ ) {
							if ( observation.getNrBins() % (binsPerBlock * binsPerThread) != 0 ) {
								continue;
							} else if ( (DMsPerThread + (2 * periodsPerThread) + binsPerThread) + (3 * periodsPerThread * binsPerThread) + (2 * DMsPerThread * periodsPerThread * binsPerThread) > maxItemsPerThread ) {
								break;
							}

							vector< unsigned int > parameters;

							parameters.push_back(DMsPerBlock);
							parameters.push_back(periodsPerBlock);
							parameters.push_back(binsPerBlock);
							parameters.push_back(DMsPerThread);
							parameters.push_back(periodsPerThread);
							parameters.push_back(binsPerThread);

							configurations.push_back(parameters);
						}
					}
				}
			}
		}
	}

	for ( vector< vector< unsigned int > >::const_iterator parameters = configurations.begin(); parameters != configurations.end(); parameters++ ) {
		double Acur = 0.0;
		double Aold = 0.0;
		double Vcur = 0.0;
		double Vold = 0.0;

		try {
			// Generate kernel
			Folding< dataType > clFold("clFold", typeName);
			clFold.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
			clFold.setObservation(&observation);
			clFold.setNrSamplesPerBin(nrSamplesPerBin);
			clFold.setNrDMsPerBlock((*parameters)[0]);
			clFold.setNrPeriodsPerBlock((*parameters)[1]);
			clFold.setNrBinsPerBlock((*parameters)[2]);
			clFold.setNrDMsPerThread((*parameters)[3]);
			clFold.setNrPeriodsPerThread((*parameters)[4]);
			clFold.setNrBinsPerThread((*parameters)[5]);
			clFold.generateCode();

			foldedData->copyHostToDevice();
			clFold(0, dedispersedData, foldedData, readCounterData, writeCounterData);
			(clFold.getTimer()).reset();
			
			for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
				foldedData->copyHostToDevice();
				clFold(0, dedispersedData, foldedData, readCounterData, writeCounterData);
				
				if ( iteration == 0 ) {
					Acur = clFold.getGFLOP() / clFold.getTimer().getLastRunTime();
				} else {
					Aold = Acur;
					Vold = Vcur;

					Acur = Aold + (((clFold.getGFLOP() / clFold.getTimer().getLastRunTime()) - Aold) / (iteration + 1));
					Vcur = Vold + (((clFold.getGFLOP() / clFold.getTimer().getLastRunTime()) - Aold) * ((clFold.getGFLOP() / clFold.getTimer().getLastRunTime()) - Acur));
				}
			}
			Vcur = sqrt(Vcur / nrIterations);

			cout << observation.getNrDMs() << " " << observation.getNrPeriods() << " " << observation.getFirstPeriod() << " " << observation.getPeriodStep() << " " << observation.getNrBins() << " " << observation.getNrSamplesPerSecond() << " " << (*parameters)[0] << " " << (*parameters)[1] << " " << (*parameters)[2] << " " << (*parameters)[3] << " " << (*parameters)[4] << " " << (*parameters)[5] << " " << setprecision(3) << Acur << " " << Vcur << " " << setprecision(6) << clFold.getTimer().getAverageTime() << " " << clFold.getTimer().getStdDev() << endl;
		} catch ( OpenCLError err ) {
			cerr << err.what() << endl;
			continue;
		}
	}

	cout << endl;

	return 0;
}
