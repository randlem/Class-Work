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
 * dumpOBVector(vector<int>* v)
 *  Ouputs (dumps) the contents of a OBVector to the screen in a standard
 *  format.
 *
 * sequenceVector(vector<int>* v)
 *  Sequence a vector in the form of 0,1,2,3,... till end()-1.
 ***********************************************************************/

// INCLUDES *************************************************************
#include <vector.h>
#include <stdexcept>
#include <iostream.h>

#include "OBVector.h"

// NAMESPACES ***********************************************************
using namespace std;

// MACROS ***************************************************************

// CLASSES **************************************************************

// PROTOTYPES ***********************************************************
void dumpVector(vector<int>* v);
void sequenceVector(vector<int>* v);

// FUNCTIONS ************************************************************
// the one, the only, the main()!
int main(int argv, char argc[]) {
    
    // variable identification and initlization
    bool error = false;            // true if error in test occured
    int test_counter = 0;          // test counter
    
    cout << "BEGIN TEST OF NORMAL FUNCTION OPERATION" << endl << endl;
    
    // -- TEST OBVector::OBVector() --
    error = false;
    cout << "Testing OBVector() with no parameters" << endl;
    cout << "Expect empty object creation" << endl;
    try {
        OBVector<int> test;
        
        if(test.empty())
            cout << "Condition after is: empty" << endl;
        else
            cout << "Condition after is: not empty" << endl;
            
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
 
   
    cout << ((error) ? "Failed" : "Passed") << endl << endl;
    test_counter++;
    // -- END TEST OBVector::OBVector() -- 
    
    // -- TEST OBVector::OBVector(size_type n) --
    error = false;
    cout << "Testing OBVector(size_type n) with parameters:\n\tn=10" << endl;
    cout << "Expect object creation size of 10" << endl;
    try {
        OBVector<int> test(10);
    
        if(test.size() != 10)
            error = true;
    
        cout << "Condition after is: test2.size() = " << test.size() << endl;        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::OBVector(size_type n) --
    
    // -- TEST OBVector::OBVector(size_type n,T& t) --
    error = false;
    cout << "Testing OBVector(size_type n,T& t) with parameters:\n\tn=10\n\tt=1" << endl;
    cout << "Expect object creation size of 10 full of 1" << endl;
    try {
        OBVector<int> test(10,1);
    
        if(test.size() != 10)
            error = true;
    
        if(*test.begin() != 1)
            error = true;
            
        cout << "Condition after is: ";
        dumpVector(&test);
        cout << endl;
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    } 
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::OBVector(size_type n,T& t) --
    
    // -- TEST OBVector::OBVector(const_iterator start, const_iterator end) --
    error = false;
    cout << "Testing OBVector(const_iterator start, const_iterator end) with parameters:\n\tstart = test1.begin()\n\tend = test1.begin()+5" << endl;
    cout << "Expect object creation size of 5 full of 1" << endl;
    try {
        OBVector<int> test1(10,1);
        OBVector<int> test2(test1.begin(),test1.begin()+5);
    
        if(test2.size() != 5)
            error = true;
    
        if(*test2.begin() != 1)
            error = true;
            
        cout << "Condition after is: ";
        dumpVector(&test2);
        cout << endl;
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::OBVector(const_iterator start, const_iterator end) --
    
    // -- TEST OBVector::OBVector(const OBVector<T> &t) --
    error = false;
    cout << "Testing OBVector(const OBVector<T> &t) with parameters:\n\tt = test1" << endl;
    cout << "Expect object copy of test1" << endl;
    try {
        OBVector<int> test1(10);
        OBVector<int> test2(test1);
        int i;
        
        if(test2.size() != test1.size())
            error = true;
    
        for(i=0; i < test1.size(); i++) {
            if(*test2.begin()+i != *test1.begin()+i)
                error = true;
        }
   
        cout << "Condition after is: ";
        dumpVector(&test2);
        cout << endl;
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::OBVector(const OBVector<T> &t) --

    // -- TEST OBVector::const_reference operator[] --
    error = false;
    cout << "Testing const_reference operator[] overload" << endl;
    cout << "Expect array of sequential numbers (0 to n-1)" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        if(test[1] != *test.begin())
            error = true;
        
        cout << "Condition after is: ";
        dumpVector(&test);
        cout << endl;
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::const_reference operator[] --

    // -- TEST OBVector::reference operator[] --
    error = false;
    cout << "Testing reference operator[] overload" << endl;
    cout << "Expect array of sequential numbers (0 to n-1) apart from the first element of 1" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test[1] = 1;
        
        if(test[1] != 1)
            error = true;
        
        cout << "Condition after is: ";
        dumpVector(&test);
        cout << endl;
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::reference operator[] --
                    
    cout << "END TEST OF NORMAL FUNCTION OPERATIONS" << endl << endl;
    
    cout << "BEGIN TEST OF ABNORMAL FUNCTION OPERATIONS" << endl << endl;
    
    // -- TEST OBVector::OBVector(size_type n) --
    error = false;
    cout << "Testing OBVector(size_type n) with parameters:\n\tn=-1" << endl;
    cout << "Expect failure" << endl;
    try {
        OBVector<int> test(-1);
    
        if(test.size() != 10)
            error = true;
    
        cout << "Condition after is: test2.size() = " << test.size() << endl;        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::OBVector(size_type n) --
    
    // -- TEST OBVector::OBVector(size_type n,T& t) --
    error = false;
    cout << "Testing OBVector(size_type n,T& t) with parameters:\n\tn=-1\n\tt=1" << endl;
    cout << "Expect failure" << endl;
    try {
        OBVector<int> test(-1,1);
    
        if(test.size() != 10)
            error = true;
    
        cout << "Condition after is: test.size() = " << test.size() << endl;        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    // -- END TEST OBVector::OBVector(size_type n,T& t) --

    // -- TEST OBVector::OBVector(const_iterator start, const_iterator end) --
    error = false;
    cout << "Testing OBVector(const_iterator start, const_iterator end) with parameters:\n\tstart=test1.end()\n\tend=test1.begin()" << endl;
    cout << "Expect failure" << endl;
    try {
        OBVector<int> test1(10);
        sequenceVector(&test1);
        OBVector<int> test2(test1.end(),test1.begin());
        
        cout << "Condition after is: ";
        dumpVector(&test2);
        cout << endl;        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::OBVector(const_iterator start, const_iterator end) --
    
        // -- TEST OBVector::OBVector(const_iterator start, const_iterator end) --
    error = false;
    cout << "Testing OBVector(const_iterator start, const_iterator end) with parameters:\n\tstart=test1.begin()\n\tend=test1.begin()" << endl;
    cout << "Expect failure" << endl;
    try {
        OBVector<int> test1(10);
        sequenceVector(&test1);
        OBVector<int> test2(test1.begin(),test1.begin());
        
        cout << "Condition after is: ";
        dumpVector(&test2);
        cout << endl;        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::OBVector(const_iterator start, const_iterator end) --
    
    // -- TEST OBVector::reference operator[] --
    error = false;
    cout << "Testing reference operator[] overload" << endl;
    cout << "Expect failure" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test[0] = 1;
        
        if(test[0] != 1)
            error = true;
        
        cout << "Condition after is: ";
        dumpVector(&test);
        cout << endl;
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::reference operator[] --
    
    // -- TEST OBVector::reference operator[] --
    error = false;
    cout << "Testing reference operator[] overload" << endl;
    cout << "Expect failure" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test[test.end() - test.begin()] = 1;
        
        if(test[test.end() - test.begin()] != 1)
            error = true;
        
        cout << "Condition after is: ";
        dumpVector(&test);
        cout << endl;
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::reference operator[] --
    
    // -- TEST OBVector::reference operator[] --
    error = false;
    cout << "Testing reference operator[] overload" << endl;
    cout << "Expect failure" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test[11] = 1;
        
        cout << "Condition after is: ";
        dumpVector(&test);
        cout << endl;
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::reference operator[] --
    
    cout << "END TEST OF ABNORMAL FUNCTION OPERATIONS" << endl << endl;
    
    cout << "BEGIN TEST OF \"SAFE\" UNAFFECTED FUNCTION OPERATIONS" << endl << endl;
            
    // -- TEST OBVector::begin() --
    error = false;
    cout << "Testing OBVector::begin()" << endl;
    cout << "Expect *test,begin() = 0" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        if((*test.begin()) != 0)
            error = true;
        
        cout << "Condition after is: *test.begin() = " << (*test.begin()) << endl;
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::begin() --
    
    // -- TEST OBVector::end() --
    error = false;
    cout << "Testing OBVector::end()" << endl;
    cout << "Expect *test.end()-1 = 9" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        if(*(test.end()-1) != 9)
            error = true;
        
        cout << "Condition after is: *(test.end()-1) = " << (*(test.end()-1)) << endl;
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::end() --
    
    // -- TEST OBVector::front() --
    error = false;
    cout << "Testing OBVector::front()" << endl;
    cout << "Expect test.front() = 0" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        if(test.front() != 0)
            error = true;
        
        cout << "Condition after is: test.front() = " << test.front() << endl;
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::front() --
    
    // -- TEST OBVector::back() --
    error = false;
    cout << "Testing OBVector::back()" << endl;
    cout << "Expect test.back() = 9" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        if(test.back() != 9)
            error = true;
        
        cout << "Condition after is: test.back() = " << test.back() << endl;
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::back() --
    
    // -- TEST OBVector::push_back() --
    error = false;
    cout << "Testing OBVector::push_back() using:\n\tn=10" << endl;
    cout << "Expect test.back() = 10" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test.push_back(10);
        
        if(test.back() != 10)
            error = true;
        
        cout << "Condition after is: test.back() = " << test.back() << endl;
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::back() --
    
    // -- TEST OBVector::pop_back() --
    error = false;
    cout << "Testing OBVector::pop_back()" << endl;
    cout << "Expect test.back() = 8" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test.pop_back();
        
        if(test.back() != 8)
            error = true;
        
        cout << "Condition after is: test.back() = " << test.back() << endl;
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::pop_back() --    
    
    // -- TEST OBVector::insert() --
    error = false;
    cout << "Testing OBVector::insert() using:\n\tpos=begin()\n\tn=1\n\tt=10" << endl;
    cout << "Expect test.front() = 10" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test.insert(test.begin(),1,10);
        
        if(test.front() != 10)
            error = true;
        
        cout << "Condition after is: test.front() = " << test.front() << endl;
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::insert() --    

    // -- TEST OBVector::erase() --
    error = false;
    cout << "Testing OBVector::erase() using:\n\tpos=begin()" << endl;
    cout << "Expect test.front() = 1" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test.erase(test.begin());
        
        if(test.front() != 1)
            error = true;
        
        cout << "Condition after is: test.front() = " << test.front() << endl;
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::erase() --

    // -- TEST OBVector::insert() --
    error = false;
    cout << "Testing OBVector::insert() using:\n\tpos=begin()\n\tn=-1\n\tt=10" << endl;
    cout << "Expect failure" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test.insert(test.begin(),-1,10);
        
        if(test.front() != 10)
            error = true;
        
        cout << "Condition after is: test.front() = " << test.front() << endl;
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::insert() --    

    // -- TEST OBVector::insert() --
    error = false;
    cout << "Testing OBVector::insert() using:\n\tpos=end()\n\tn=1\n\tt=10" << endl;
    cout << "Expect failure" << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test.insert(test.end(),1,10);
        
        if(test.front() != 10)
            error = true;
        
        cout << "Condition after is: test.front() = " << test.front() << endl;
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::insert() --    
    
    // -- TEST OBVector::insert() --
    error = false;
    cout << "Testing OBVector::insert() using:\n\tpos=end()\n\tstart=test2.end()\n\tend=test2.begin()" << endl;
    cout << "Expect failure" << endl;
    try {
        OBVector<int> test(10);
        OBVector<int> test2(5,1);
        sequenceVector(&test);
        
        test.insert(test.begin(),test2.end(),test2.begin());
        
        if(test.front() != 1)
            error = true;
        
        cout << "Condition after is: ";
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::insert() --
    
    // -- TEST OBVector::erase() --
    error = false;
    cout << "Testing OBVector::erase() using:\n\tpos=end()" << endl;
    cout << "Expect failure." << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test.erase(test.end());
        
        if(test.front() != 1)
            error = true;
        
        cout << "Condition after is: ";
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::erase() --
    
    // -- TEST OBVector::erase() --
    error = false;
    cout << "Testing OBVector::erase() using:\n\tstart=end()\n\tfinish=begin()" << endl;
    cout << "Expect failure." << endl;
    try {
        OBVector<int> test(10);
        sequenceVector(&test);
        
        test.erase(test.end(),test.begin());
        
        if(test.size() != 0)
            error = true;
        
        cout << "Condition after is: ";
        dumpVector(&test);
        cout << endl;
        
    }
    catch ( const exception & e ) {
        cout << "Error: " << e.what() << endl;
        error = true;
    }
        
    cout << ((error) ? "Failed" : "Passed") << endl << endl;    
    test_counter++;
    // -- END TEST OBVector::erase() --
    
    cout << "END TEST OF \"SAFE\" UNAFFECTED FUNCTION OPERATIONS" << endl << endl;
     
    // do a little diagonistic output
    cout << "Number of test cases run: " << test_counter << endl;
    
    // exit back to the os
    exit(0);

}// end main()

void dumpVector(vector<int>* v) {
    // This function dumps the contents of a OBVector to the screen
    // in a standard format.
    
    // variable identification and initlziation
    vector<int>::iterator i;
    
    for(i=v->begin(); i!=v->end(); i++) {
        cout << *i << " ";
    }    
    
}

void sequenceVector(vector<int>* v) {
    // This function takes an exsisting vector and sequences the 
    // elements in it. Something like:
    // 0,1,2,3,4,5,6,end-1
    
    // variable identification and initlziation
    vector<int>::iterator i;
    
    for(i=v->begin(); i!=v->end(); i++) {
        *i = i - v->begin();
    }    

}
