/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

SimPlus.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of SimPlus,
see SimPlus.h.

****************************************************************************/

#include "SimPlus.h"

#include <iostream>

using std::endl;
using std::cerr;

SimPlus* SimPlus::theKernel = 0;

SimPlus::SimPlus()
{
	simTime = 0.0;
	theEventList = new EventHeap;
}

SimPlus::~SimPlus()
{
	while( !randomStack.empty() )
	{
		delete randomStack.top();
		randomStack.pop();
	}
	while( !sampleStack.empty() )
	{
		delete sampleStack.top();
		sampleStack.pop();
	}
	while( !queueStack.empty() )
	{
		delete queueStack.top();
		queueStack.pop();
	}
	delete theEventList;
}

SimPlus* SimPlus::getInstance()
{
	if( theKernel == 0 )
		theKernel = new SimPlus();
	return theKernel;
}

void SimPlus::registerServer( ServerEntity* aServer )
{
	serverMap.insert( std::make_pair( aServer->getID(), aServer ) );
}

void SimPlus::reportError( const string& theError )
{
	cerr << endl << "************SimPlus Kernel Panic***************" << endl;
	cerr << theError << endl;
	cerr << "*************Unable to Continue****************" << endl << endl;
	delete theKernel;
	exit( 1 );
}

Event* SimPlus::getEvent()
{
	return theEventPool.get();
}

void SimPlus::releaseEvent( Event* releaseMe )
{
	theEventPool.release( releaseMe );
}

Event* SimPlus::timing()
{
	Event* temp = theEventList->get();
	simTime = temp->getTimeStamp();

	map<unsigned int, ServerEntity*>::iterator theIterator;

	theIterator = serverMap.find( temp->getDestination() );
	if( theIterator == serverMap.end() )
		return temp;
	theIterator->second->processEvent( temp );
	return 0;
}

void SimPlus::scheduleEvent( Event* scheduleMe )
{
	if( !theEventList->put( scheduleMe ) )
	{
		reportError( "Event List has exceeded system memory limits for the current user." );
	}
}

bool SimPlus::cancelEventType( const unsigned short& eventType )
{
	return theEventList->cancelNext( eventType );
}

bool SimPlus::cancelEventID( const unsigned int& eventID )
{
	return false;
}


ExponentialDist* SimPlus::getExponentialDist( const double& theta,
	const RNGFactory::RNGType type)
{
	ExponentialDist* pTempRNG = new ExponentialDist( theta, type );
	randomStack.push( (RawDist*) pTempRNG );
	return pTempRNG;
}

NormalDist* SimPlus::getNormalDist( const double& mu, const double& sigma,
	const RNGFactory::RNGType type )
{
	NormalDist* pTempRNG = new NormalDist( mu, sigma, type );
	randomStack.push( (RawDist*) pTempRNG );
	return pTempRNG;
}

UniformDist* SimPlus::getUniformDist( const double& lowerBound,
	const double& upperBound, const RNGFactory::RNGType type )
{
	UniformDist* pTempRNG = new UniformDist( lowerBound, upperBound, type );
	randomStack.push( (RawDist*) pTempRNG );
	return pTempRNG;
}

SampST* SimPlus::getSampST()
{
	SampST* pTempStat = new SampST;
	sampleStack.push( pTempStat );
	return pTempStat;
}


EntityQueue* SimPlus::getEntityQueue()
{
	EntityQueue* pTempQueue = new EntityQueue;
	queueStack.push( pTempQueue );
	return pTempQueue;
}

void SimPlus::expandEventPool( const unsigned short& newEvents )
{
	theEventPool.reserve( newEvents );
}

unsigned int SimPlus::availableEvents()
{
	return theEventPool.getSize();
}
