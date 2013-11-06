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
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <CLData.hpp>
#include <utils.hpp>
#include <Folding.hpp>
#include <Timer.hpp>
using isa::utils::ArgumentList;
using isa::utils::toStringValue;
using isa::utils::Timer;
using AstroData::Observation;
using isa::OpenCL::initializeOpenCL;
using isa::OpenCL::CLData;
using PulsarSearch::Folding;

typedef float dataType;
const string typeName("float");
const unsigned int maxThreadsPerBlock = 1024;
const unsigned int maxThreadsMultiplier = 512;
const unsigned int maxItemsPerThread = 256;
const unsigned int maxItemsMultiplier = 256;
const unsigned int padding = 32;

// Common parameters
const unsigned int nrBeams = 1;
const unsigned int nrStations = 64;
// LOFAR
/*const float minFreq = 138.965f;
const float channelBandwidth = 0.195f;
const unsigned int nrSamplesPerSecond = 200000;
const unsigned int nrChannels = 32;*/
// Apertif
const float minFreq = 1425.0f;
const float channelBandwidth = 0.2929f;
const unsigned int nrSamplesPerSecond = 20000;
const unsigned int nrChannels = 1024;
// Periods
const unsigned int nrBins = 256;


int main(int argc, char * argv[]) {
	unsigned int nrDMs = 0;
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	Observation< dataType > observation("FoldingTuning", typeName);
	CLData< dataType > * dedispersedData = new CLData< dataType >("DedispersedData", true);
	CLData< dataType > * foldedData = new CLData<dataType >("FoldedData", true);
	CLData< unsigned int > * counterData = new CLData< unsigned int >("CounterData", true);


	try {
		ArgumentList args(argc, argv);

		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		nrDMs = args.getSwitchArgument< unsigned int >("-dms");
	} catch ( exception & err ) {
		cerr << err.what() << endl;
		return 1;
	}

	// Setup of the observation
	observation.setPadding(padding);
	observation.setNrSamplesPerSecond(nrSamplesPerSecond);
	observation.setNrDMs(nrDMs);
	observation.setFirstPeriod(nrBins);
	observation.setPeriodStep(nrBins);
	observation.setNrBins(nrBins);
	
	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();
	
	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	cout << fixed << endl;
	cout << "# nrDMs nrPeriods nrDMsPerBlock nrPeriodsPerBlock nrBinsPerBlock nrDMsPerThread nrPeriodsPerThread nrBinsPerThread GFLOP/s err time err" << endl << endl;

	for ( unsigned int nrPeriods = 2; nrPeriods <= 1024; nrPeriods *= 2 ) {
		observation.setNrPeriods(nrPeriods);

		// Allocate memory
		dedispersedData->allocateHostData(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
		foldedData->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
		foldedData->blankHostData();
		counterData->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
		counterData->blankHostData();

		dedispersedData->setCLContext(clContext);
		dedispersedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
		foldedData->setCLContext(clContext);
		foldedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
		counterData->setCLContext(clContext);
		counterData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));

		try {
			dedispersedData->allocateDeviceData();
			foldedData->allocateDeviceData();
			foldedData->copyHostToDevice();
			counterData->allocateDeviceData();
			counterData->copyHostToDevice();
		} catch ( OpenCLError err ) {
			cerr << err.what() << endl;
		}

		// Find the parameters
		vector< unsigned int > DMsPerBlock;
		for ( unsigned int DMs = 2; DMs <= maxThreadsPerBlock; DMs++ ) {
			if ( (observation.getNrDMs() % DMs) == 0 ) {
				DMsPerBlock.push_back(DMs);
			}
		}

		for ( vector< unsigned int >::iterator DMs = DMsPerBlock.begin(); DMs != DMsPerBlock.end(); DMs++ ) {
			for (unsigned int periodsPerBlock = 1; periodsPerBlock <= maxThreadsMultiplier; periodsPerBlock++ ) {
				if ( (*DMs * periodsPerBlock) > maxThreadsPerBlock ) {
					break;
				} else if ( (observation.getNrPeriods() % periodsPerBlock) != 0 ) {
					continue;
				}

				for ( unsigned int binsPerBlock = 1; binsPerBlock <= maxThreadsMultiplier; binsPerBlock++ ) {
					if ( (*DMs * periodsPerBlock * binsPerBlock) > maxThreadsPerBlock ) {
						break;
					} else if ( (observation.getNrBins() % binsPerBlock) != 0 ) {
						continue;
					}

					for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
						if ( (observation.getNrDMs() % (*DMs * DMsPerThread)) != 0 ) {
							continue;
						}

						for ( unsigned int periodsPerThread = 1; periodsPerThread <= maxItemsPerThread; periodsPerThread++ ) {
							if ( (DMsPerThread * periodsPerThread) > maxItemsPerThread ) {
								break;
							} else if ( (observation.getNrPeriods() % (periodsPerBlock * periodsPerThread)) != 0 ) {
								continue;
							}

							for ( unsigned int binsPerThread = 1; binsPerThread <= maxItemsPerThread; binsPerThread++ ) {
								if ( (DMsPerThread * periodsPerThread * binsPerThread) > maxItemsPerThread ) {
									break;
								} else if ( (observation.getNrBins() % (binsPerBlock * binsPerThread)) != 0 ) {
									continue;
								}

								double Acur = 0.0;
								double Aold = 0.0;
								double Vcur = 0.0;
								double Vold = 0.0;

								try {
									// Generate kernel
									Folding< dataType > clFold("clFold", typeName);
									clFold.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
									clFold.setObservation(&observation);
									clFold.setNrDMsPerBlock(*DMs);
									clFold.setNrPeriodsPerBlock(periodsPerBlock);
									clFold.setNrBinsPerBlock(binsPerBlock);
									clFold.setNrDMsPerThread(DMsPerThread);
									clFold.setNrPeriodsPerThread(periodsPerThread);
									clFold.setNrBinsPerThread(binsPerThread);
									clFold.generateCode();

									clFold(dedispersedData, foldedData, counterData);
									(clFold.getTimer()).reset();
									for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
										clFold(dedispersedData, foldedData, counterData);
										
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

									cout << nrDMs << " " << nrPeriods << " " << *DMs << " " << periodsPerBlock << " " << binsPerBlock << " " << DMsPerThread << " " << periodsPerThread << " " << binsPerThread << " " << setprecision(3) << Acur << " " << Vcur << " " << setprecision(6) << clFold.getTimer().getAverageTime() << " " << clFold.getTimer().getStdDev() << endl;
								} catch ( OpenCLError err ) {
									cerr << err.what() << endl;
									continue;
								}
							}
						}
					}
				}
			}
		}
		cout << endl << endl;
	}

	cout << endl;

	return 0;
}
