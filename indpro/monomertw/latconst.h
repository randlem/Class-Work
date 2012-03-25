#ifndef LATCONST_H
#define LATCONST_H

// lattice dimensions and area
//const int DIM_X = 4096;
//const int DIM_X = 2048;
const int DIM_X = 1024;                      // the total X dimension of the lattice
//const int DIM_X = 512;                      // the total X dimension of the lattice
//const int DIM_Y = 1024;                       // the total Y dimension of the lattice
const int DIM_Y = 256;
const int GHOST = 1;                         // the size of the ghost region
const int LEFT_X_BOUNDRY = GHOST;            // the left boundry
const int RIGHT_X_BOUNDRY = DIM_X - GHOST - 1;   // the right boundry
const int SIZE = DIM_X * DIM_Y;              // the area (number) of sites in the lattice

// enviroment attributes
const int NUM_DIR = 8; // number of directions

// minimum local time
const double MINIMUM_TIME = 0.001;

// random number vars
const int NUMBER_START = 10000; // the number of random numbers to start with
const int SEED = 0;             // the initial seed

#endif

