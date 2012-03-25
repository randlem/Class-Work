#include <iostream>
using std::ostream;
using std::endl;

#include <string>
using std::string;

#include "SimPlus.h"
#include "Object.h"

#ifndef SERVER_H
#define SERVER_H

class Server : public ServerEntity {
	public:
		Server( double, double, EntityQueue* );
		~Server();

		virtual Event* generateEvent( const unsigned short&, const double& );
		virtual void processEvent( Event* );

		int getNumProcessed() { return(numProcessed); }

		friend ostream& operator<<( ostream&, Server& );

	protected:
		Object* currentObject;
		int numProcessed;
		SampST* idleTimeStat;
		SampST* waitTimeStat;
		SampST* serviceTimeStat;
		NormalDist* myServiceTime;
		double lastStart;
		double lastStop;

	private:
};

#endif
