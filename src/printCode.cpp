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
#include <Observation.hpp>
#include <utils.hpp>
#include <Folding.hpp>


int main(int argc, char *argv[]) {
  std::string typeName;
  PulsarSearch::FoldingConf conf;
	AstroData::Observation observation;

	try {
    isa::utils::ArgumentList args(argc, argv);
    typeName = args.getSwitchArgument< std::string >("-type");
    observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    conf.setVector(args.getSwitchArgument< unsigned int >("-vector"));
    conf.setNrDMsPerBlock(args.getSwitchArgument< unsigned int >("-db"));
    conf.setNrPeriodsPerBlock(args.getSwitchArgument< unsigned int >("-pb"));
    conf.setNrBinsPerBlock(args.getSwitchArgument< unsigned int >("-bb"));
    conf.setNrDMsPerThread(args.getSwitchArgument< unsigned int >("-dt"));
    conf.setNrPeriodsPerThread(args.getSwitchArgument< unsigned int >("-pt"));
    conf.setNrBinsPerThread(args.getSwitchArgument< unsigned int >("-bt"));
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
    observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
    observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), args.getSwitchArgument< unsigned int >("-first_period"), args.getSwitchArgument< unsigned int >("-period_step"));
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
	} catch  ( isa::utils::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " -type ... -padding ... -vector ... -db ... -pb ... -bb ... -dt ... -pt ... -bt ... -samples ... -dms ... -periods ... -bins ... -first_period ... -period_step ..." << std::endl;
		return 1;
	}

  // Generate kernel
  std::string * code = PulsarSearch::getFoldingOpenCL(conf, typeName, observation);
  std::cout << *code << std::endl;

	return 0;
}

