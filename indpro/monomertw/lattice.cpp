#include "lattice.h"

Lattice::Lattice() : localTime(MINIMUM_TIME), minGlobalTime(0.0),
					 depositionRate(1.0), diffusionRate(1.0e6),
					 countDiffusion(0), countEvents(0), countBoundry(0), countRemote(0), countRollback(0), rng(10000,7) {

	lattice = new site*[DIM_X+GHOST+GHOST];

	// set up the lattice array
	for(int i=0; i < DIM_X + GHOST + GHOST; i++) {
		lattice[i] = new site[DIM_Y];

		for(int j=0; j < DIM_Y; j++) {
			lattice[i][j].p.x = i;
			lattice[i][j].p.y = j;
			lattice[i][j].h = 0;
			lattice[i][j].listIndex = -1;
		}
	}

	// set the directions
	movementDir[0].y= 1;  movementDir[0].x= 0;
	movementDir[1].y= 0;  movementDir[1].x= 1;
	movementDir[2].y=-1;  movementDir[2].x= 0;
	movementDir[3].y= 0;  movementDir[3].x=-1;
	movementDir[4].y= 1;  movementDir[4].x= 1;
	movementDir[5].y= 1;  movementDir[5].x=-1;
	movementDir[6].y=-1;  movementDir[6].x= 1;
	movementDir[7].y=-1;  movementDir[7].x=-1;
}

Lattice::~Lattice() {
	for(int i=0; i < DIM_X + GHOST + GHOST; ++i) {
		delete [] lattice[i];
	}
	delete [] lattice;
}

void Lattice::cleanup(fstream& logFile) {
	vector<message> m;

	logFile << ((mpi.isMessage()) ? "messages waiting" : "message queue empty") << endl;
	logFile << ((mpi.isAntiMessage()) ? "messages waiting" : "message queue empty") << endl;

	if(mpi.isMessage()) {
		mpi.recvMessages(&m);
		m.empty();
	}

	if(mpi.isAntiMessage()) {
		mpi.recvAntiMessages(&m);
		m.empty();
	}
}

bool Lattice::doNextEvent() {
	double nextKMCTime = computeTime();
	EventType nextEventType;

	if(remoteEventList.empty() || nextKMCTime < remoteEventList.top()->time) {
		// get the event type
		nextEventType = getNextEventType();

		// depending on the event type commit different events
		switch(nextEventType) {
			case eventDeposition: {
				point p;

				// set up the new site object
				p.x = (int)(rng.getRandom(localTime) * (DIM_X));
				p.y = (int)(rng.getRandom(localTime) * DIM_Y);

				// commit the event
				commitEvent(new Event(&lattice[p.x][p.y],localTime,true,nextEventType));
			} break;
			case eventDiffusion: {
				site *oldSite, *newSite;
				int index = (int)(rng.getRandom(localTime) * monomerList.size());

				// make sure there actually a monomer in the system to diffuse
				if(monomerList.size() > 0) {
					// set the old site (random monomer from the list) and get a new site
					oldSite = monomerList[index];
					newSite = randomMove(oldSite);
					monomerList.add(newSite,localTime);
					monomerList.remove(index,localTime);


					// set the index of the new site and update our entry in the monomerList
					newSite->listIndex = index;
					monomerList[index] = newSite;

					// clear the listIndex of the old site
					oldSite->listIndex = -1;

					// commit the event
					commitEvent(new Event(oldSite,newSite,localTime,true,nextEventType,index));

				}
			} break;
		}

		// set the local time to the nextKMCTime
		localTime = nextKMCTime;

	} else {
		// see if there is a waiting anti-event for this remote event
		if(hasAntiEvent(remoteEventList.top())) {
			// delete the remote event to play nice with memory
			delete remoteEventList.top();

			// remove the remote event from the queue
			remoteEventList.pop();

			// return false abandoning trying to do an event this cycle
			return(false);
		}

		// commit the remote event to the simulation
		commitEvent(remoteEventList.top());

		// set nextKMCTime to the remote time
		localTime = remoteEventList.top()->time;

		// remove the remote event from the queue
		remoteEventList.pop();

		countRemote++;
	}

	// incriment the event counter
	++countEvents;

	return(true);
}

double Lattice::computeTime() {
	double Drate = diffusionRate * monomerList.size() * 0.25f;
	double totaldep = depositionRate * SIZE;
	double dt = -log(rng.getRandom(localTime))/(Drate+totaldep);

	return(localTime + dt);
}

EventType Lattice::getNextEventType() {
	float Drate = 0.25f * monomerList.size() * diffusionRate;
	float Trate = Drate + (depositionRate * SIZE);
	float prob = (Drate / Trate);

	// if the next random number from the stream is less then the probality
	// the the next event is a eventDiffusion, return a diffusion event
	if(rng.getRandom(localTime) < prob)
	   return(eventDiffusion);

	// catch-all is the deposition event
	return(eventDeposition);
}

bool Lattice::commitEvent(Event* event) {

	if(event == NULL)
		throw(Exception("NULL event passed to Lattice::commitEvent()!"));

	// process the event based on the event type
	switch(event->eventType) {
		case eventDeposition: {
			// DEBUG:
			// file << "deposition: (" << event->newSite->p.x << "," << event->newSite->p.y << ") " << event->newSite->h << endl;

			// incriment the height up
			++(event->newSite->h);

			// see if the monomer falls on the boundry (x == LEFT_X_BOUNDRY || x == RIGHT_X_BOUNDRY)
			if(isBoundry(event->newSite->p) && event->isLocal) {
				// send the event off to the correct neighbor
				mpi.sendMessage(makeMessage(event),GET_DIR(event->newSite->p.x));

				// set the boundry event flag
				event->isBoundry = true;

				// incriment the countBoundry
				++countBoundry;
			}

			// see if the monomer will bond
			if(!isBound(event->newSite)) {
				// no bond was formed so move on
				event->newSite->listIndex = monomerList.add(event->newSite,event->time);
			}

		} break;
		case eventDiffusion: {
			// DEBUG
			// file << "diffusion (" << event->oldSite->p.x << "," << event->oldSite->p.y << ") " << event->oldSite->h
			//	 << " => (" << event->newSite->p.x << "," << event->newSite->p.y << ") " << event->newSite->h << endl;

			// incriment the new site height up
			++(event->newSite->h);

			// incriment the old site height down
			--(event->oldSite->h);

			// see if the monomers new site falls on the boundry (x == 1 || x == DIM - 1)
			if(isBoundry(event->newSite->p) && event->isLocal) {
				// send the event off to the correct neighbor
				mpi.sendMessage(makeMessage(event),GET_DIR(event->newSite->p.x));

				// set the boundry event flag
				event->isBoundry = true;

				// incriment the countBoundry
				++countBoundry;
			}

			// see if the monomer will bond
			if(isBound(event->newSite)) {
				// remove the current monomer from the monomer list
				site* s = monomerList.remove(event->newSite->listIndex,event->time);

				// because of the way RewindList works remove() returns the value of
				// the new element at the position, and we'll need to change it's
				// listIndex value to reflect it's new index
				if(s != NULL)
					s->listIndex = event->newSite->listIndex;

				// invalidate the index of the newSite
				event->newSite->listIndex = -1;

				// clear any neighbors that could be unbound monomers from the list
				clearBonded(event->newSite,event->time);
			}

			// incriment the diffusion counter
			++countDiffusion;
		} break;
		default: throw(Exception("commitEvent(): Invalid Event type!"));
	}

	// push the event into the event list
	eventList.push(event);

	return(true);
}

site* Lattice::randomMove(site* oldSite) {
	point newPoint;
	int i = (int)(rng.getRandom(localTime) * 4);

	// randomly move in a random direction with help from our direction array
	newPoint.x = oldSite->p.x + movementDir[i].x;
	newPoint.y = oldSite->p.y + movementDir[i].y;

	// make sure we don't leave our domain
	if(newPoint.x >= RIGHT_X_BOUNDRY)
		newPoint.x -= 2;
	if(newPoint.x <= LEFT_X_BOUNDRY)
		newPoint.x += 2;

	if(newPoint.y >= DIM_Y)
		newPoint.y -= 2;
	if(newPoint.y < 0)
		newPoint.y += 2;

	// return the new site on the lattice (where the monomer moved to)
	return(&lattice[newPoint.x][newPoint.y]);
}

bool Lattice::isBoundry(point p) {
	// if p.x falls in the ghost (0, DIM-1) or boundry (1, DIM-2) return true
	if(p.x < LEFT_X_BOUNDRY || p.x > RIGHT_X_BOUNDRY)
		return(true);

	// default is false
	return(false);
}

bool Lattice::isBound(site* s) {
	point p;

	// loop and check each direction, returning true if a neighbor is at the
	// same height or higher
	for(int i=0; i < NUM_DIR; ++i) {
		p = s->p;
		p.x += movementDir[i].x;
		p.y += movementDir[i].y;

		if(p.x >= 0 && p.x < DIM_X+GHOST+GHOST && p.y >= 0 && p.y < DIM_Y)
			if(lattice[p.x][p.y].h >= s->h)
				return(true);
	}

	// default is false
	return(false);
}

bool Lattice::clearBonded(site* s, const double t) {
	point p;

	// loop and inspect the neighbors marking any bonded and
	for(int i=0; i < NUM_DIR; ++i) {
		p = s->p;
		p.x += movementDir[i].x;
		p.y += movementDir[i].y;

		// if the point is a valid point and if the point is going to bound,
		// and if the listIndex is valid then delete it from the monomer list
		if(p.x >= RIGHT_X_BOUNDRY && p.x < RIGHT_X_BOUNDRY && p.y >= 0 && p.y < DIM_Y)
			if(lattice[p.x][p.y].h >= s->h)
				if(lattice[p.x][p.y].listIndex != -1) {
					monomerList.remove(lattice[p.x][p.y].listIndex,t);
					lattice[p.x][p.y].listIndex = -1;
				}
	}

	return(true);
}

bool Lattice::createHeightMap(string filename) {
	int x, y;
    int width=DIM_X, height=DIM_Y;
    png_byte color_type=PNG_COLOR_TYPE_RGBA;
    png_byte bit_depth=8;
    png_structp png_ptr;
    png_infop info_ptr;
    int number_of_passes=1;
    png_bytep * row_pointers;
    FILE* fp;

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
        row_pointers[y] = (png_byte*) malloc(width*((bit_depth/8)*4));

    for (y=0; y<height; y++) {
        png_byte* row = row_pointers[y];
        for (x=0; x<width; x++) {
            png_byte* ptr = &(row[x*4]);
            if(lattice[x][y].h > 0) {
                ptr[0] = 255 - ((lattice[x][y].h*10)%255); ptr[1] = 255 - ((lattice[x][y].h*10)%255); ptr[2] = 255 - ((lattice[x][y].h*10)%255); ptr[3] = 255;
            } else {
                ptr[0] = 0; ptr[1] = 0; ptr[2] = 0; ptr[3] = 255;
            }
        }
    }

    fp = fopen(filename.c_str(), "wb");
    if(fp == NULL) {
		throw Exception("Couldn't open height map file!");
		return(false);
    } else {
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        info_ptr = png_create_info_struct(png_ptr);
        png_init_io(png_ptr, fp);
        png_set_IHDR(png_ptr, info_ptr, width, height,
                 bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
        png_write_info(png_ptr, info_ptr);
        png_write_image(png_ptr, row_pointers);
        png_write_end(png_ptr, NULL);
    }

	return(true);
}

bool Lattice::rollback(double t) {
	Event* event;

	if(t < minGlobalTime)
		t = minGlobalTime;

	// rewind the lattice using the eventList stack
	while(!eventList.empty() && localTime > t) {
		event = eventList.top();

		switch(event->eventType) {
			case eventDeposition: {
				// decrease the height of the deposition site
				--event->newSite->h;

				// clear the listIndex just in case
				event->newSite->listIndex = -1;

				// see if the monomers new site falls on the boundry (x == 1 || x == DIM - 1)
				if(isBoundry(event->newSite->p)) {
					if(event->isLocal) {
						// send the anti-message
						mpi.sendAntiMessage(makeMessage(event),GET_DIR(event->newSite->p.x));
					} else {
						// store the event back in the remoteEvents list
						remoteEventList.push(event);
						countRemote--;
					}

					// decriment the countBoundry
					--countBoundry;
				}

			} break;
			case eventDiffusion: {
				// decrease the height of the new site
				--event->newSite->h;

				// increase the height of the old site
				++event->oldSite->h;

				// clear the listIndex just in case
				event->newSite->listIndex = -1;
				event->oldSite->listIndex = -1;

				// see if the monomers new site falls on the boundry (x == 1 || x == DIM - 1)
				if(isBoundry(event->newSite->p)) {
					if(event->isLocal) {
						// send the anti-message
						mpi.sendAntiMessage(makeMessage(event),GET_DIR(event->newSite->p.x));
					} else {
						// store the event back in the remoteEvents list
						remoteEventList.push(event);
						countRemote--;
					}

					// decriment the countBoundry
					--countBoundry;
				}

				// decriment the countDiffusion stat
				--countDiffusion;
			} break;
		}

		// set the local clock to the event time
		localTime = event->time;

		// decriment the countEvents;
		--countEvents;

		// if it's a local event clean up the memory
		if(event->isLocal)
			delete event;

		// pop the top of the event list
		eventList.pop();
	}

	// rewind the RNG
	rng.rewind(t);

	// we generate 3 rngs for the wrong cycle before we advance the clock, so
	// this is a work-a-round to fix that
	rng.rewind(3);

	// rewind the monomerList
	monomerList.rewind(t);

	// fix all of the listIndex entries
	for(int i=0; i < monomerList.size(); ++i) {
		monomerList[i]->listIndex = i;
	}

	++countRollback;

	return(true);
}

bool Lattice::negoitateEvents(fstream& logFile) {
	vector<message> messages;
	vector<message> antiMessages;
	vector<Event*> remoteEvents;
	float pastTime = 0.0;
	bool isRollback = false;

	// get any waiting message
//	logFile << "lattice.mpi.recvMessages()" << endl; logFile.flush();
	mpi.recvMessages(&messages);

	// get any waiting anti-messages
//	logFile << "lattice.mpi.recvAntiMessages()" << endl; logFile.flush();
	mpi.recvAntiMessages(&antiMessages);

	// process waiting antimessages
	if(!antiMessages.empty()) {
//		logFile << "processing anti-messages!" << endl; logFile.flush();
		// get the lowest time of any past antimessages
		for(vector<message>::iterator i=antiMessages.begin(); i < antiMessages.end(); ++i) {
			if((*i).time <= localTime) {
				if((*i).time < pastTime || !isRollback) {
					pastTime = (*i).time;
					isRollback = true;
				}
			}
		}

		// if there is a past event rollback to the minimum past event type
		if(isRollback) {
			// rollback to the time of this antimessage
			rollback(pastTime);
		}

		// translate the antimessages and insert them into the antiEvent vector
		translateMessages(&antiEvents,&antiMessages);
	}

	// if we don't have any waiting messages just exit out
	if(messages.empty())
		return(true);

//	logFile << "processing messages!" << endl; logFile.flush();

	// translate the remote messages into events4
	translateMessages(&remoteEvents,&messages);

	// loop through all the recieved events and push them into the
	isRollback = false; pastTime = 0.0;
	for(vector<Event*>::iterator i=remoteEvents.begin(); i < remoteEvents.end(); ++i) {
		if((*i)->time < localTime) {
			if((*i)->time < pastTime || !isRollback) {
				pastTime = (*i)->time;
				isRollback = true;
			}
		}
	}

	// if we need to rollback then do a rollback
	if(isRollback)
		rollback(pastTime);

	// push all the future events onto the remote event list
	for(vector<Event*>::iterator i=remoteEvents.begin(); i < remoteEvents.end(); ++i)
		remoteEventList.push(*i);

	return(true);
}

bool Lattice::translateMessages(vector<Event*>* events, vector<message>* messages) {
	vector<message>::iterator i;
	message m;

	for(i=messages->begin(); i < messages->end(); ++i) {
		m = *i;

		// make an event from it <sarcasm>this should be fun</sarcasm>
		switch(m.type) {
			case eventDiffusion: {
				// translate the oldSite and newSite coords
				if(m.newSite.p.x < LEFT_X_BOUNDRY) {
					m.newSite.p.x = DIM_X - 1 - m.newSite.p.x;
					m.oldSite.p.x = DIM_X - 1 - m.oldSite.p.x;
				} else {
					m.newSite.p.x -= DIM_X - 1;
					m.oldSite.p.x -= DIM_X - 1;
				}

				// push an event into the return vector
				events->push_back(new Event(&lattice[m.oldSite.p.x][m.oldSite.p.y],
				                           &lattice[m.newSite.p.x][m.newSite.p.y],m.time,false,m.type,0));
			} break;
			case eventDeposition: {
				if(m.newSite.p.x < LEFT_X_BOUNDRY)
					m.newSite.p.x = DIM_X - 1 - m.newSite.p.x;
				else
					m.newSite.p.x -= DIM_X - 1;

				// push an event into the return vector
				events->push_back(new Event(&lattice[m.newSite.p.x][m.newSite.p.y],m.time,false,m.type));
			} break;
			default:throw(Exception("Bad event type encountered in Lattice::translateMessages()"));
		}
	}

	return(true);
}

message* Lattice::makeMessage(Event* event) {

	switch(event->eventType) {
		case eventDiffusion: {
			m.oldSite.p.x = event->oldSite->p.x;
			m.oldSite.p.y = event->oldSite->p.y;
			m.oldSite.h = event->oldSite->h;
			//m.oldSite = *(event->oldSite);
		} // fall through
		case eventDeposition: {
			m.newSite.p.x = event->newSite->p.x;
			m.newSite.p.y = event->newSite->p.y;
			m.newSite.h = event->newSite->h;
			//m.newSite = *(event->newSite);
		}break;
	}
	m.time = event->time;
	m.type = event->eventType;

	return(&m);
}

bool Lattice::hasAntiEvent(Event* event) {

	// if there are no antiEvents just bail
	if(antiEvents.empty())
		return(false);

	// loop through the avaliable antiEvents
	for(vector<Event*>::iterator i=antiEvents.begin(); i < antiEvents.end(); ++i) {
		// if the time of the event is the same as the time of the antievent
		// we can return true and erase the antievent
		if(event->time == (*i)->time) {
			delete *i; // clean up the allocated memeory
			antiEvents.erase(i);
			return(true);
		}
	}

	// no matching antievent was found to return false
	return(false);
}
