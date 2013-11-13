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
	inline void setNrDMsPerBlock(unsigned int DM);
	inline void setNrSamplesPerBlock(unsigned int samples);

	inline void setObservation(Observation< T > * obs);

private:
	unsigned int nrThreadsPerBlock;
	unsigned int nrDMsPerBlock;
	unsigned int nrSamplesPerBlock;
	cl::NDRange globalSize;
	cl::NDRange localSize;

	Observation< T > * observation;
};


// Implementation
template< typename T > Transpose< T >::Transpose(string name, string dataType) : Kernel< T >(name, dataType), nrThreadsPerBlock(0), nrDMsPerBlock(0), nrSamplesPerBlock(0), globalSize(cl::NDRange(1, 1, 1)), localSize(cl::NDRange(1, 1, 1)), observation(0) {}

template< typename T > void Transpose< T >::generateCode() throw (OpenCLError) {
	// Begin kernel's template
	string localElements_s = toStringValue< unsigned int >(nrDMsPerBlock * nrSamplesPerBlock);
	string nrThreadsPerBlock_s = toStringValue< unsigned int >(nrThreadsPerBlock);
	string nrDMsPerBlock_s = toStringValue< unsigned int >(nrDMsPerBlock);
	string nrSamplesPerBlock_s = toStringValue< unsigned int >(nrSamplesPerBlock);
	string nrSamplesPerPaddedSecond_s = toStringValue< unsigned int >(observation->getNrSamplesPerPaddedSecond());
	string nrPaddedDMs_s = toStringValue< unsigned int >(observation->getNrPaddedDMs());

	delete this->code;
	this->code = new string();
	*(this->code) = "__kernel void Transpose(__global const " + this->dataType + " * const restrict input, __global " + this->dataType + " * const restrict output) {\n"
	"unsigned int baseDM = get_group_id(0) * " + nrDMsPerBlock_s + ";\n"
	"unsigned int baseSample = (get_group_id(1) * " + nrSamplesPerBlock_s + ") + get_local_id(0);\n"
	"__local "+ this->dataType + " tempStorage[" + localElements_s + "];"
	"\n"
	"for ( unsigned int DM = baseDM; DM < baseDM + " + nrDMsPerBlock_s + "; DM++ ) {\n"
	"for ( unsigned int sample = baseSample; sample < baseSample + " + nrSamplesPerBlock_s + "; sample += " + nrThreadsPerBlock_s + " ) {\n"
	"tempStorage[((DM - baseDM) * " + nrSamplesPerBlock_s + ") + ( sample - baseSample)] = input[(DM * " + nrSamplesPerPaddedSecond_s + ") + sample];\n"
	"}\n"
	"}\n"
	"barrier(CLK_LOCAL_MEM_FENCE);\n"
	"baseSample = (get_group_id(1) * " + nrSamplesPerBlock_s + ");\n"
	"baseDM = get_group_id(0) * " + nrDMsPerBlock_s + " + get_local_id(0);\n"
	"for ( unsigned int sample = baseSample; sample < baseSample + " + nrSamplesPerBlock_s + "; sample++ ) {\n"
	"for ( unsigned int DM = baseDM; DM < baseDM + " + nrDMsPerBlock_s + "; DM += " + nrThreadsPerBlock_s + " ) {\n"
	"output[(sample * " + nrPaddedDMs_s + ") + DM] = tempStorage[((DM - baseDM) * " + nrSamplesPerBlock_s + ") + ( sample - baseSample)];"
	"}\n"
	"}\n"
	"}\n";
	// End kernel's template

	globalSize = cl::NDRange(observation->getNrDMs() / nrDMsPerBlock, observation->getNrSamplesPerSecond() / nrSamplesPerBlock);
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

template< typename T > inline void Transpose< T >::setNrDMsPerBlock(unsigned int DMs) {
	nrDMsPerBlock = DMs;
}

template< typename T > inline void Transpose< T >::setNrSamplesPerBlock(unsigned int samples) {
	nrSamplesPerBlock = samples;
}

} // PulsarSearch

#endif // TRANSPOSE_HPP
