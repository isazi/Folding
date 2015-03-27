// Copyright 2015 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <Folding.hpp>

namespace PulsarSearch {

void readTunedFoldingConf(tunedFoldingConf & tunedFolding, const std::string & foldingFilename) {
	std::string temp;
	std::ifstream foldingFile(foldingFilename);

	while ( ! foldingFile.eof() ) {
		unsigned int splitPoint = 0;

		std::getline(foldingFile, temp);
		if ( ! std::isalpha(temp[0]) ) {
			continue;
		}
		std::string deviceName;
		unsigned int nrDMs = 0;
		unsigned int nrPeriods = 0;
    PulsarSearch::FoldingConf parameters;

		splitPoint = temp.find(" ");
		deviceName = temp.substr(0, splitPoint);
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		nrDMs = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		nrPeriods = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		parameters.setNrDMsPerBlock(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		parameters.setNrPeriodsPerBlock(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		parameters.setNrBinsPerBlock(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		parameters.setNrDMsPerThread(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		parameters.setNrPeriodsPerThread(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		parameters.setNrBinsPerThread(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
		temp = temp.substr(splitPoint + 1);
		parameters.setVector(isa::utils::castToType< std::string, unsigned int >(temp));

		if ( tunedFolding.count(deviceName) == 0 ) {
      std::map< unsigned int, std::map< unsigned int, PulsarSearch::FoldingConf > > externalContainer;
      std::map< unsigned int, PulsarSearch::FoldingConf > internalContainer;

			internalContainer.insert(std::make_pair(nrPeriods, parameters));
			externalContainer.insert(std::make_pair(nrDMs, internalContainer));
			tunedFolding.insert(std::make_pair(deviceName, externalContainer));
		} else if ( tunedFolding[deviceName].count(nrDMs) == 0 ) {
      std::map< unsigned int, PulsarSearch::FoldingConf > internalContainer;

			internalContainer.insert(std::make_pair(nrPeriods, parameters));
			tunedFolding[deviceName].insert(std::make_pair(nrDMs, internalContainer));
		} else {
			tunedFolding[deviceName][nrDMs].insert(std::make_pair(nrPeriods, parameters));
		}
	}
}

FoldingConf::FoldingConf() {}

FoldingConf::~FoldingConf() {}

std::string FoldingConf::print() const {
  return std::string(isa::utils::toString(nrDMsPerBlock) + " " + isa::utils::toString(nrPeriodsPerBlock) + " " + isa::utils::toString(nrBinsPerBlock) + " " + isa::utils::toString(nrDMsPerThread) + " " + isa::utils::toString(nrPeriodsPerThread) + " " + isa::utils::toString(nrBinsPerThread) + " " + isa::utils::toString(vector));
}

std::string * getFoldingOpenCL(const FoldingConf & conf, const std::string & dataType, const AstroData::Observation & observation) {
  std::string * code = new std::string();
	std::string nrSamplesPerSecond_s = isa::utils::toString< unsigned int >(observation.getNrSamplesPerSecond());
	std::string nrPaddedDMs_s  = isa::utils::toString< unsigned int >(observation.getNrPaddedDMs());
	std::string firstPeriod_s = isa::utils::toString< unsigned int >(observation.getFirstPeriod());
	std::string periodStep_s = isa::utils::toString< unsigned int >(observation.getPeriodStep());
	std::string nrPaddedBins_s = isa::utils::toString< unsigned int >(observation.getNrPaddedBins());

	// Begin kernel's template
  *code = "__kernel void folding(const unsigned int second, __global const " + dataType + " * const restrict samples, __global " + dataType + " * const restrict bins, __global const unsigned int * const restrict readCounters, __global unsigned int * const restrict writeCounters, __global const unsigned int * const restrict nrSamplesPerBin) {\n"
    "unsigned int sample = 0;\n"
    "<%DEFS_PERIOD%>"
    "<%DEFS_BIN%>"
    "<%DEFS_DM%>"
    "<%DEFS_PERIOD_BIN%>"
    "<%DEFS_PERIOD_BIN_DM%>"
    "\n"
    "<%COMPUTE%>"
    "}\n";
	std::string defsPeriodTemplate = "const unsigned int period<%PERIOD_NUM%> = (get_group_id(1) * " + isa::utils::toString(conf.getNrPeriodsPerBlock() * conf.getNrPeriodsPerThread()) + ") + get_local_id(1) + <%PERIOD_OFFSET%>;\n"
    "const unsigned int period<%PERIOD_NUM%>Value = " + firstPeriod_s + " + (period<%PERIOD_NUM%> * " + periodStep_s + ");\n";
	std::string defsBinTemplate = "const unsigned int bin<%BIN_NUM%> = (get_group_id(2) * " + isa::utils::toString(conf.getNrBinsPerBlock() * conf.getNrBinsPerThread()) + ") + get_local_id(2) + <%BIN_OFFSET%>;\n";
	std::string defsDMTemplate;
  if ( conf.getVector() == 1 ) {
    defsDMTemplate = "const unsigned int DM<%DM_NUM%> = (get_group_id(0) * " + isa::utils::toString(conf.getNrDMsPerBlock() * conf.getNrDMsPerThread()) + ") + get_local_id(0) + <%DM_OFFSET%>;\n";
  } else {
    defsDMTemplate = "const unsigned int DM<%DM_NUM%> = (get_group_id(0) * " + isa::utils::toString(conf.getNrDMsPerBlock() * conf.getNrDMsPerThread() * conf.getVector()) + ") + (get_local_id(0) * " + isa::utils::toString(conf.getVector()) + ") + <%DM_OFFSET%>;\n";
  }
	std::string defsPeriodBinTemplate = "const unsigned int samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%> = nrSamplesPerBin[(period<%PERIOD_NUM%> * " + isa::utils::toString(observation.getNrBins() * isa::utils::pad(2, observation.getPadding())) + ") + (bin<%BIN_NUM%> * " + isa::utils::toString(isa::utils::pad(2, observation.getPadding())) + ")];\n"
		"const unsigned int offsetp<%PERIOD_NUM%>b<%BIN_NUM%> = nrSamplesPerBin[(period<%PERIOD_NUM%> * " + isa::utils::toString(observation.getNrBins() * isa::utils::pad(2, observation.getPadding())) + ") + (bin<%BIN_NUM%> * " + isa::utils::toString(isa::utils::pad(2, observation.getPadding())) + ") + 1];\n"
		"const unsigned int pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> = readCounters[(period<%PERIOD_NUM%> * " + nrPaddedBins_s + ") + bin<%BIN_NUM%>];\n"
    "unsigned int foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%> = 0;\n";
  std::string defsPeriodBinDMTemplate;
  if ( conf.getVector() == 1 ) {
    defsPeriodBinDMTemplate = dataType + " foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%> = 0;\n";
  } else {
    defsPeriodBinDMTemplate = dataType + isa::utils::toString(conf.getVector()) + " foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%> = 0;\n";
  }
	std::string computeTemplate = "if ( samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%> > 0 ) {\n"
    "sample = offsetp<%PERIOD_NUM%>b<%BIN_NUM%> + ((pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> / samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%>) * period<%PERIOD_NUM%>Value) + (pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> % samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%>);\n"
    "if ( (sample / "+ nrSamplesPerSecond_s + ") == second ) {\n"
    "sample %= "+ nrSamplesPerSecond_s + ";\n"
    "}\n"
    "while ( sample < " + nrSamplesPerSecond_s + " ) {\n"
    "<%COMPUTE_DM%>"
    "foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>++;\n"
    "if ( ((foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%> + pCounterp<%PERIOD_NUM%>b<%BIN_NUM%>) % samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%>) == 0 ) {\n"
    "sample += period<%PERIOD_NUM%>Value - (samplesPerBinp<%PERIOD_NUM%>b<%BIN_NUM%> - 1);\n"
    "} else {\n"
    "sample++;\n"
    "}\n"
    "}\n"
    "if ( foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%> > 0 ) {\n"
    "unsigned int outputItem = (bin<%BIN_NUM%> * " + isa::utils::toString(observation.getNrPeriods() * observation.getNrPaddedDMs()) + ") + (period<%PERIOD_NUM%> * " + nrPaddedDMs_s + ");\n"
    + dataType + " pValue = 0;\n"
    "<%STORE_DM%>"
    "}\n"
    "}\n"
    "if ( get_local_id(0) == 0) {\n"
    "writeCounters[(period<%PERIOD_NUM%> * " + nrPaddedBins_s + ") + bin<%BIN_NUM%>] = pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> + foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>;\n"
    "}\n";
  std::string computeDMTemplate;
  if ( conf.getVector() == 1 ) {
    computeDMTemplate += "foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%> += samples[(sample * " + nrPaddedDMs_s + ") + DM<%DM_NUM%>];\n";
  } else {
    computeDMTemplate += "foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%> += vload" + isa::utils::toString(conf.getVector()) + "(0, &(samples[(sample * " + nrPaddedDMs_s + ") + DM<%DM_NUM%>]));\n";
  }
  std::string storeDMTemplate = "pValue = bins[outputItem + DM<%DM_NUM%>];\n";
  if ( conf.getVector() == 1 ) {
    storeDMTemplate += "bins[outputItem + DM<%DM_NUM%>] = ((pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> * pValue) + (foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%>)) / (pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> + foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>);\n";
  } else {
    storeDMTemplate += "vstore" + isa::utils::toString(conf.getVector()) + "(((pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> * pValue) + (foldedSamplep<%PERIOD_NUM%>b<%BIN_NUM%>d<%DM_NUM%>)) / (pCounterp<%PERIOD_NUM%>b<%BIN_NUM%> + foldedCounterp<%PERIOD_NUM%>b<%BIN_NUM%>), 0, &(bins[outputItem + DM<%DM_NUM%>]));\n";
  }
	// End kernel's template

	std::string * defsPeriod_s = new std::string();
  std::string * defsBin_s = new std::string();
  std::string * defsDM_s = new std::string();
  std::string * defsPeriodBin_s = new std::string();
  std::string * defsPeriodBinDM_s = new std::string();
	std::string * compute_s = new std::string();

  for ( unsigned int bin = 0; bin < conf.getNrBinsPerThread(); bin++ ) {
    std::string bin_s = isa::utils::toString< unsigned int >(bin);
    std::string offset_s = isa::utils::toString(bin * conf.getNrBinsPerBlock());
    std::string * temp = 0;

    temp = isa::utils::replace(&defsBinTemplate, "<%BIN_NUM%>", bin_s);
    if ( bin == 0 ) {
      std::string empty_s;
      temp = isa::utils::replace(temp, " + <%BIN_OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%BIN_OFFSET%>", offset_s, true);
    }
    defsBin_s->append(*temp);
    delete temp;
  }
  for ( unsigned int dm = 0; dm < conf.getNrDMsPerThread(); dm++ ) {
    std::string dm_s = isa::utils::toString< unsigned int >(dm);
    std::string offset_s = isa::utils::toString(dm * conf.getNrDMsPerBlock() * conf.getVector());
    std::string * temp = 0;

    temp = isa::utils::replace(&defsDMTemplate, "<%DM_NUM%>", dm_s);
    if ( dm == 0 ) {
      std::string empty_s;
      temp = isa::utils::replace(temp, " + <%DM_OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%DM_OFFSET%>", offset_s, true);
    }
    defsDM_s->append(*temp);
    delete temp;
  }
  for ( unsigned int period = 0; period < conf.getNrPeriodsPerThread(); period++ ) {
    std::string period_s = isa::utils::toString< unsigned int >(period);
    std::string offset_s = isa::utils::toString(period * conf.getNrPeriodsPerBlock());
    std::string * temp = 0;

    temp = isa::utils::replace(&defsPeriodTemplate, "<%PERIOD_NUM%>", period_s);
    if ( period == 0 ) {
      std::string empty_s;
      temp = isa::utils::replace(temp, " + <%PERIOD_OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%PERIOD_OFFSET%>", offset_s, true);
    }
    defsPeriod_s->append(*temp);
    delete temp;

    for ( unsigned int bin = 0; bin < conf.getNrBinsPerThread(); bin++ ) {
      std::string bin_s = isa::utils::toString< unsigned int >(bin);
      std::string * temp = 0;
      std::string * computeDM_s = new std::string();
      std::string * storeDM_s = new std::string();

      temp = isa::utils::replace(&defsPeriodBinTemplate, "<%PERIOD_NUM%>", period_s);
      temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
      defsPeriodBin_s->append(*temp);
      delete temp;

      for ( unsigned int dm = 0; dm < conf.getNrDMsPerThread(); dm++ ) {
        std::string dm_s = isa::utils::toString< unsigned int >(dm);
        std::string * temp = 0;

        temp = isa::utils::replace(&defsPeriodBinDMTemplate, "<%PERIOD_NUM%>", period_s);
        temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
        temp = isa::utils::replace(temp, "<%DM_NUM%>", dm_s, true);
        defsPeriodBinDM_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&computeDMTemplate, "<%PERIOD_NUM%>", period_s);
        temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
        temp = isa::utils::replace(temp, "<%DM_NUM%>", dm_s, true);
        computeDM_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&storeDMTemplate, "<%PERIOD_NUM%>", period_s);
        temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
        temp = isa::utils::replace(temp, "<%DM_NUM%>", dm_s, true);
        storeDM_s->append(*temp);
        delete temp;
      }
      temp = isa::utils::replace(&computeTemplate, "<%PERIOD_NUM%>", period_s);
      temp = isa::utils::replace(temp, "<%BIN_NUM%>", bin_s, true);
      temp = isa::utils::replace(temp, "<%COMPUTE_DM%>", *computeDM_s, true);
      temp = isa::utils::replace(temp, "<%STORE_DM%>", *storeDM_s, true);
      compute_s->append(*temp);
      delete temp;
    }
  }
  code = isa::utils::replace(code, "<%DEFS_PERIOD%>", *defsPeriod_s, true);
  code = isa::utils::replace(code, "<%DEFS_BIN%>", *defsBin_s, true);
  code = isa::utils::replace(code, "<%DEFS_DM%>", *defsDM_s, true);
  code = isa::utils::replace(code, "<%DEFS_PERIOD_BIN%>", *defsPeriodBin_s, true);
  code = isa::utils::replace(code, "<%DEFS_PERIOD_BIN_DM%>", *defsPeriodBinDM_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);

  return code;
}

} // PulsarSearch

