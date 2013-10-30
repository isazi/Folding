/*
 * Copyright (C) 2012
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
#include <ReadData.hpp>
#include <InitializeOpenCL.hpp>
#include <GPUData.hpp>
#include <utils.hpp>
#include <Shifts.hpp>
#include <Dedispersion.hpp>
#include <Folding.hpp>
#include <kernels/Memset.hpp>
using isa::utils::ArgumentList;
using isa::utils::same;
using AstroData::Observation;
using AstroData::readLOFAR;
using isa::OpenCL::initializeOpenCL;
using isa::OpenCL::GPUData;
using isa::OpenCL::Memset;
using PulsarSearch::getShifts;
using PulsarSearch::dedisperse;
using PulsarSearch::fold;
using PulsarSearch::Folding;

typedef float dataType;
const string typeName("float");
const unsigned int nrBins = 256;


int main(int argc, char *argv[]) {
	unsigned int firstSecond = 0;
	unsigned int nrSeconds = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	long long unsigned int wrongValues = 0;
	long long unsigned int wrongCounts = 0;
	string headerFileName;
	string dataFileName;
	Observation< dataType > observation("FoldingTest", typeName);

	try {
		ArgumentList args(argc, argv);

		firstSecond = args.getSwitchArgument< unsigned int >("-fs");
		nrSeconds = args.getSwitchArgument< unsigned int >("-ns");
		
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");

		headerFileName = args.getSwitchArgument< string >("-header");
		dataFileName = args.getSwitchArgument< string >("-data");

		observation.setFirstDM(args.getSwitchArgument< float >("-dm_first"));
		observation.setDMStep(args.getSwitchArgument< float >("-dm_step"));
		observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dm_number"));
		observation.setFirstPeriod(args.getSwitchArgument< unsigned int >("-period_first"));
		observation.setPeriodStep(args.getSwitchArgument< unsigned int >("-period_step"));
		observation.setNrPeriods(args.getSwitchArgument< unsigned int >("-period_number"));
		observation.setNrBins(nrBins);
	}
	catch ( exception &err ) {
		cerr << err.what() << endl;
		return 1;
	}
	
	// Load the observation data
	vector< GPUData< dataType > * > *input = new vector< GPUData< dataType > * >(1);
	readLOFAR(headerFileName, dataFileName, observation, *input, nrSeconds, firstSecond);

	// Print some statistics
	cout << fixed << setprecision(3) << endl;
	cout << "Total seconds: \t\t" << observation.getNrSeconds() << endl;
	cout << "Min frequency: \t\t" << observation.getMinFreq() << " MHz" << endl;
	cout << "Max frequency: \t\t" << observation.getMaxFreq() << " MHz" << endl;
	cout << "Nr. channels: \t\t" << observation.getNrChannels() << endl;
	cout << "Channel bandwidth: \t" << observation.getChannelBandwidth() << " MHz" << endl;
	cout << "Samples/second: \t" << observation.getNrSamplesPerSecond() << endl;
	cout << "Min sample: \t\t" << observation.getMinValue() << endl;
	cout << "Max sample: \t\t" << observation.getMaxValue() << endl;
	cout << endl;

	// Test
	cl::Context *clContext = new cl::Context();
	vector< cl::Platform > *clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > *clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > *clQueues = new vector< vector < cl::CommandQueue > >();
	
	unsigned int nrSamplesPerChannel = 0;
	unsigned int secondsToBuffer = 0;
	GPUData< unsigned int > *shifts = getShifts(observation);
	GPUData< unsigned int > *globalCount = new GPUData< unsigned int >("GlobalCount", true);
	GPUData< unsigned int > *clGlobalCount = new GPUData< unsigned int >("CLGlobalCount", true);
	GPUData< dataType > *dispersedData = new GPUData< dataType >("DispersedData", true, true);
	GPUData< dataType > *dedispersedData = new GPUData< dataType >("DedispersedData", true);
	GPUData< dataType > *foldedData = new GPUData< dataType >("FoldedData", true);
	GPUData< dataType > *clFoldedData = new GPUData< dataType >("CLFoldedData", true);

	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);
	
	if ( ((observation.getNrSamplesPerSecond() + (*shifts)[((observation.getNrDMs() - 1) * observation.getNrPaddedChannels())]) % 4) != 0 ) {
		nrSamplesPerChannel = (observation.getNrSamplesPerSecond() + (*shifts)[((observation.getNrDMs() - 1) * observation.getNrPaddedChannels())]) + (4 - ((observation.getNrSamplesPerSecond() + (*shifts)[((observation.getNrDMs() - 1) * observation.getNrPaddedChannels())]) % 4));
	}
	else {
		nrSamplesPerChannel = (observation.getNrSamplesPerSecond() + (*shifts)[((observation.getNrDMs() - 1) * observation.getNrPaddedChannels())]);
	}
	secondsToBuffer = static_cast< unsigned int >(ceil(static_cast< float >(nrSamplesPerChannel) / observation.getNrSamplesPerPaddedSecond()));
	if ( nrSeconds < secondsToBuffer ) {
		cerr << "Not enough seconds." << endl;
		return 1;
	}
	
	// Allocate memory
	globalCount->allocateHostData(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
	clGlobalCount->allocateHostData(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
	clGlobalCount->setCLContext(clContext);
	clGlobalCount->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	clGlobalCount->allocateDeviceData();
	dispersedData->allocateHostData(secondsToBuffer * observation.getNrChannels() * observation.getNrSamplesPerPaddedSecond());
	dedispersedData->allocateHostData(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond());
	dedispersedData->setCLContext(clContext);
	dedispersedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	dedispersedData->allocateDeviceData();
	foldedData->allocateHostData(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
	clFoldedData->allocateHostData(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
	clFoldedData->setCLContext(clContext);
	clFoldedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	clFoldedData->allocateDeviceData();

	// Generate kernel
	Memset< dataType > *memsetDT = new Memset< dataType>(typeName);
	memsetDT->bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
	memsetDT->setNrThreads(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
	memsetDT->setNrRows(observation.getNrDMs() * observation.getNrPeriods());
	memsetDT->setNrThreadsPerBlock(observation.getNrBins());
	memsetDT->generateCode();
	(*memsetDT)(static_cast< dataType >(0), clFoldedData);
	delete memsetDT;
	Memset< unsigned int > *memsetUI = new Memset< unsigned int >("unsigned int");
	memsetUI->bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
	memsetUI->setNrThreads(observation.getNrDMs() * observation.getNrPeriods() * observation.getNrPaddedBins());
	memsetUI->setNrRows(observation.getNrDMs() * observation.getNrPeriods());
	memsetUI->setNrThreadsPerBlock(observation.getNrBins());
	memsetUI->generateCode();
	(*memsetUI)(0, clGlobalCount);
	delete memsetUI;
	Folding< dataType > clFold("clFold", typeName);
	clFold.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
	clFold.setObservation(&observation);
	clFold.setNrPeriodsPerBlock(256);
	clFold.setNrPeriodsPerThread(1);
	clFold.generateCode();
	
	cout << clFold.getCode() << endl;
	cout << endl;

	// Dedispersion and Folding loop
	for ( unsigned int second = 0; second < nrSeconds - (secondsToBuffer - 1); second++ ) {
		for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
			for ( unsigned int chunk = 0; chunk < secondsToBuffer; chunk++ ) {
				memcpy(dispersedData->getRawHostDataAt((channel * secondsToBuffer * observation.getNrSamplesPerPaddedSecond()) + (chunk * observation.getNrSamplesPerSecond())), (input->at(second + chunk))->getRawHostDataAt(channel * observation.getNrSamplesPerPaddedSecond()), observation.getNrSamplesPerSecond() * sizeof(dataType));
			}
		}

		dedisperse(secondsToBuffer * observation.getNrSamplesPerPaddedSecond(), observation, dispersedData, dedispersedData, shifts);
		fold(second, observation, dedispersedData, foldedData, globalCount);
		
		dedispersedData->copyHostToDevice();
		clFold(second, dedispersedData, clFoldedData, clGlobalCount);
	}

	clFoldedData->copyDeviceToHost();
	clGlobalCount->copyDeviceToHost();
	for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
		for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
			for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
				if ( !same((*foldedData)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin], (*clFoldedData)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin]) ) {
					wrongValues++;

					cout << dm << " " << period << " " << bin << " " << (*foldedData)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin] << " " << (*clFoldedData)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin] << " " << (*foldedData)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin] - (*clFoldedData)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin] << endl;
				}
				if ( (*globalCount)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin] != (*clGlobalCount)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin] ) {
					wrongCounts++;

					cout << dm << " " << period << " " << bin << " " << (*globalCount)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin] << " " << (*clGlobalCount)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin] << " " << static_cast< int >((*globalCount)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin]) - static_cast< int >((*clGlobalCount)[(((dm * observation.getNrPeriods()) + period) * observation.getNrPaddedBins()) + bin]) << endl;
				}

			}
		}
	}
	cout << endl;

	cout << "Wrong samples: " << wrongValues << " (" << (wrongValues * 100) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins()) << "%)." << endl;
	cout << "Wrong counters: " << wrongCounts << " (" << (wrongCounts * 100) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins()) << "%)." << endl;
	cout << endl;
	cout << fixed << setprecision(6);
	cout << "Kernel timing: " << clFold.getTime() << " (total), " << clFold.getTime() / (nrSeconds - (secondsToBuffer - 1)) << " (average)" << endl;
	cout << endl;

	return 0;
}

