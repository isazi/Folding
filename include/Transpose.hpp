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

#include <Kernel.hpp>
using isa::OpenCL::Kernel;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <Observation.hpp>
using AstroData::Observation;
#include <utils.hpp>
using isa::utils::toStringValue;
using isa::utils::giga;


#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

namespace PulsarSearch {

// OpenCL transpose
template< typename T > class Transpose : public Kernel< T > {
public:
	Transpose(string name, string dataType);

	void generateCode() throw (OpenCLError);
	void operator()(CLData< T > * input, CLData< T > * output) throw (OpenCLError);

	inline void setNrThreadsPerBlock(unsigned int threads);

	inline void setObservation(Observation< T > * obs);

private:
	unsigned int nrThreadsPerBlock;
	cl::NDRange globalSize;
	cl::NDRange localSize;

	Observation< T > * observation;
};


// Implementation
template< typename T > Transpose< T >::Transpose(string name, string dataType) : Kernel< T >(name, dataType), nrThreadsPerBlock(0), globalSize(cl::NDRange(1, 1, 1)), localSize(cl::NDRange(1, 1, 1)), observation(0) {}

template< typename T > void Transpose< T >::generateCode() throw (OpenCLError) {
	// Begin kernel's template
	string localElements_s = toStringValue< unsigned int >(nrThreadsPerBlock * nrThreadsPerBlock);
	string nrThreadsPerBlock_s = toStringValue< unsigned int >(nrThreadsPerBlock);
	string nrSamplesPerPaddedSecond_s = toStringValue< unsigned int >(observation->getNrSamplesPerPaddedSecond());
	string nrSamplesPerSecond_s = toStringValue< unsigned int >(observation->getNrSamplesPerSecond());
	string nrPaddedDMs_s = toStringValue< unsigned int >(observation->getNrPaddedDMs());

	delete this->code;
	this->code = new string();
	*(this->code) = "__kernel void " + this->name + "(__global const " + this->dataType + " * const restrict input, __global " + this->dataType + " * const restrict output) {\n"
	"const unsigned int baseDM = get_group_id(0) * " + nrThreadsPerBlock_s + ";\n"
	"const unsigned int baseSample = get_group_id(1) * " + nrThreadsPerBlock_s + ";\n"
	"__local "+ this->dataType + " tempStorage[" + localElements_s + "];"
	"\n"
	// Load input
	"for ( unsigned int DM = 0; DM < " + nrThreadsPerBlock_s + "; DM++ ) {\n"
	"if ( baseSample + get_local_id(0) < " + nrSamplesPerSecond_s + " ) {\n"
	"tempStorage[(DM * " + nrThreadsPerBlock_s + ") + get_local_id(0)] = input[((baseDM + DM) * " + nrSamplesPerPaddedSecond_s + ") + (baseSample + get_local_id(0))];\n"
	"}\n"
	"}\n"
	"barrier(CLK_LOCAL_MEM_FENCE);\n"
	// Local in-place transpose
	"for ( unsigned int i = 1; i <= " + nrThreadsPerBlock_s + " / 2; i++ ) {\n"
	"unsigned int localItem = (get_local_id(0) + i) % " + nrThreadsPerBlock_s + ";\n"
	+ this->dataType + " temp = 0;\n"
	"if ( (i != "+ nrThreadsPerBlock_s + ") || (get_local_id(0) < " + nrThreadsPerBlock_s + " / 2) ) {\n"
	"temp = tempStorage[(get_local_id(0) * " + nrThreadsPerBlock_s + ") + localItem];\n"
	"tempStorage[(get_local_id(0) * " + nrThreadsPerBlock_s + ") + localItem] = tempStorage[(localItem * " + nrThreadsPerBlock_s + ") + get_local_id(0)];\n"
	"tempStorage[(localItem * " + nrThreadsPerBlock_s + ") + get_local_id(0)] = temp;\n"
	"}\n"
	"}\n"
	"barrier(CLK_LOCAL_MEM_FENCE);\n"
	// Store output
	"for ( unsigned int sample = 0; sample < " + nrThreadsPerBlock_s + "; sample++ ) {\n"
	"if ( baseSample + sample < " + nrSamplesPerSecond_s + " ) {\n"
	"output[((baseSample + sample) * " + nrPaddedDMs_s + ") + (baseDM + get_local_id(0))] = tempStorage[(sample * " + nrThreadsPerBlock_s + ") + get_local_id(0)];"
	"}\n"
	"}\n"
	"}\n";
	// End kernel's template

	globalSize = cl::NDRange(observation->getNrDMs(), ceil(observation->getNrSamplesPerSecond() / nrThreadsPerBlock));
	localSize = cl::NDRange(nrThreadsPerBlock, 1);

	this->gb = giga(static_cast< long long unsigned int >(observation->getNrDMs()) * observation->getNrSamplesPerSecond() * 2 * sizeof(T));

	this->compile();
}

template< typename T > void Transpose< T >::operator()(CLData< T > * input, CLData< T > * output) throw (OpenCLError) {
	this->setArgument(0, *(input->getDeviceData()));
	this->setArgument(1, *(output->getDeviceData()));

	this->run(globalSize, localSize);
}

template< typename T > inline void Transpose< T >::setNrThreadsPerBlock(unsigned int threads) {
	nrThreadsPerBlock = threads;
}

template< typename T > inline void Transpose< T >::setObservation(Observation< T > * obs) {
	observation = obs;
}

} // PulsarSearch

#endif // TRANSPOSE_HPP
