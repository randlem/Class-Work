#include <iostream>
using std::ostream;
using std::endl;

#include "SimPlus.h"

#ifndef OBJECT_H
#define OBJECT_H

class Object : public Entity {
	public:
		Object();
		~Object();

		static void setEntryQueue( EntityQueue* );

		void beginWait( double );
		double endWait( double );
		void enterSystem( double );
		void exitSystem( double );

		static void getStats(ostream&);

	protected:
		static SampST* totalTimeStat;

		double entryTime;
		double startedWaiting;

	private:
};

#endif
