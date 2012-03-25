/************************************************************************
 * FILE:       main.cpp
 * WRITTEN BY: Mark Randles
 * COURSE:     CS335
 * ASSIGNMENT: Lab Assignment 1
 * DUE DATE:   11:59PM Saturday, Sept 13
 *
 * OVERVIEW: A simple wrapper class designed to create a one based array 
 *  (visual basic style).  All functions that could be affected by the 
 *  wrapper have been overridden.  Also a test program has been created 
 *  to test as many fail conditions as possible.
 *
 * INPUT: No user input.
 *
 * OUTPUT: Pass or fail test messages.
 *
 * FUNCTIONS:
 * 
 * main(int argv, char* argc[])
 *  Main entry point for the program.  This function also makes all the
 *  the calls to the new derived array class for the tests.
 * 
 * dumpOBVector(OBVector* v)
 *  Ouputs (dumps) the contents of a OBVector to the screen in a standard
 *  format.
 *
 ***********************************************************************/

// INCLUDES *************************************************************
#include <vector.h>
#include <stdexcept>
#include <iostream.h>

#include "OBVector.h"

// NAMESPACES ***********************************************************
using std::vector;

// MACROS ***************************************************************

// CLASSES **************************************************************

// PROTOTYPES ***********************************************************
void dumpOBVector(OBVector<int>* v);

// FUNCTIONS ************************************************************
// the one, the only, the main()!
int main(int argv, char argc[]) {
    
    // variable identification and initlization
    OBVector<int> obvector1;
    OBVector<int> obvector2(10);
    OBVector<int> obvector3(10,1);
    vector<int>::iterator i;
    int j;
    
    cout << "Test the default allocation routines." << endl;
    cout << "\tobvector1.size() = " << obvector1.size() << endl;
    cout << "\tobvector2.size() = " << obvector2.size() << endl;
    cout << "\tobvector3.size() = " << obvector3.size() << endl;
    
    cout << "Test the default fill constructor." << endl;
    cout << "\tobvector3[5] = " << obvector3[5] << endl;
    
    cout << "Test the begin() and end() function overrides." << endl;
    j = 0;
    for(i=obvector2.begin(); i!=obvector2.end(); i++) {
        j++;
    }
    cout << "\tobvector2.size() = " << obvector2.size() << endl;
    cout << "\tj                = " << j << endl;
    
    cout << "Fill the vector with sequential data." << endl;
    j=0;
    for(i=obvector2.begin(); i!=obvector2.end(); i++) {
        j++;
        *i = j;
        cout << "\tobvector2[" << j << "] = " << *i << endl;
    }
    
    cout << "Test the [] override." << endl;
    cout << "\tobvector2[1] = " << obvector2[1] << endl;
    cout << "\tobvector2[10] = " << obvector2[10] << endl;
    
    cout << "Test the erase(first,last) override." << endl;
    obvector3.erase(obvector3.begin()+1,obvector3.end()-1);
    dumpOBVector(&obvector3);
    
    cout << "Test the erase(position) override." << endl;
    obvector3.erase(obvector3.begin());
    dumpOBVector(&obvector3);
    
    cout << "Test the insert() overriders." << endl;
    obvector3.insert(obvector3.begin(),5);
    dumpOBVector(&obvector3);
    cout << "\tsize = " << obvector3.size() << endl;
    cout << endl;
    obvector3.insert(obvector3.begin()+2);
    dumpOBVector(&obvector3);
    cout << "\tsize = " << obvector3.size() << endl;
    cout << "obvector[1]" << obvector3[1] << endl;
    
    // exit back to the os
    exit(0);

}// end main()

void dumpOBVector(OBVector<int>* v)
{
    // This function dumps the contents of a OBVector to the screen
    // in a standard format.
    
    // variable identification and initlziation
    int j=1;
    vector<int>::iterator i;
    
    for(i=v->begin(); i!=v->end(); i++) {
        cout << "\tv[" << j << "] = " << *i << endl;
        j++;
    }    
    
}
