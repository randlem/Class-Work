/************************************************************************
 * FILE:       OBVector.h
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
 * CLASSES:
 * 
 * class OBVector
 *  Override of the STL Vector class.  
 *
 *  OBVector()
 *   Default constructor.  Call the matching Vector constructor.
 *
 *  OBVector(size_type n)
 *   Default constructor.  Call the matching Vector Constructor.
 *
 *  OBVector(size_type n, const T& t)
 *   Default constructor.  Call the matching Vector constructor.
 *  
 *  OBVector(int n, const T& t)
 *   Default constructor.  Call the matching Vector constructor.
 *
 *  OBVector(long n, const T& t)
 *   Default constructor.  Call the matching Vector constructor.
 * 
 *  OBVector(const_iterator start, const_iterator end)
 *   Default constructor.  Call the matching Vector constructor.
 *  
 *  OBVector(const OBVector<T> &t)
 *   Copy constructor.  Call the vector copy constructor.
 * 
 *  ~OBVector()
 *   Destructor.  Call the vector destructor.
 *  
 *  refrence operator[] (size_type n)
 *   Overload the [] operator to address elements in the vector.
 *  
 *  const_refrence operator[] (size_type n)
 *   Overload the [] operator to address elements in the vector.
 *
 *  iterator erase(iterator first, iterator last)
 *   Erase the part of the vector defined by the set [first, last].
 *
 *  iterator erase(iterator position)
 *   Erase the element of the vector at position
 * 
 *  iterator insert(iterator position, const T &x)
 *   Insert a new elemnt into the vector.
 * 
 *  iterator insert(iterator position)
 *   Insert a new element into the vector.
 * 
 *  void insert(iterator position, const_iterator first, const_iterator last)
 *   Insert a new element into the vector.
 * 
 *  void insert(iterator pos, size_type n, const T &x)
 *   Insert a new element into the vector.
 *
 *  void insert(iterator pos, int n, const T &x)
 *   Insert a new element into the vector.
 *
 *  void insert(iterator pos, long n, const T &x)
 *   Insert a new element into the vector. 
 *
 ***********************************************************************/

#ifndef _OBVECTOR_H_
#define _OBVECTOR_H_

// INCLUDES *************************************************************
#include <vector.h>
#include <stdexcept>
#include <iostream.h>
#include <exception>

// NAMESPACES ***********************************************************
using std::vector;
using std::out_of_range;
using std::invalid_argument;

// MACROS ***************************************************************

// CLASSES **************************************************************
template <typename T>
class OBVector : public vector<T> {
    public:
    
        OBVector() : vector<T>() { }
        OBVector(size_type n) : vector<T>() {
            // THIS HACK IS NOT NESSARY, IF BGUNIX HAD UP-TO-DATE STL LIBRARIES
            // THIS WOULD NOT BE A PROBLEM.
            if(((signed int)n) < 0)
                throw invalid_argument("Invalid Size");
                
            clear();
            resize(n);
            
        }
        
        OBVector(size_type n, const T& t) : vector<T>() {
            // THIS HACK IS NOT NESSARY, IF BGUNIX HAD UP-TO-DATE STL LIBRARIES
            // THIS WOULD NOT BE A PROBLEM.
            if(((signed int)n) < 0)
                throw invalid_argument("Invalid Size");
                
            clear();                
            resize(n,t);
        
        }
        
        OBVector(int n, const T& t) : vector<T>() {
            // THIS HACK IS NOT NESSARY, IF BGUNIX HAD UP-TO-DATE STL LIBRARIES
            // THIS WOULD NOT BE A PROBLEM.
            if(((signed int)n) < 0)
                throw invalid_argument("Invalid Size");
            
            clear();
            resize(n,t);
        
        }
        
        OBVector(long n, const T& t) : vector<T>() {
            // THIS HACK IS NOT NESSARY, IF BGUNIX HAD UP-TO-DATE STL LIBRARIES
            // THIS WOULD NOT BE A PROBLEM.
            if(((signed int)n) < 0)
                throw invalid_argument("Invalid Size");
                
            clear();
            resize(n,t);
        
        }
        
        OBVector(const_iterator start, const_iterator end) : vector<T>() {
            // THIS HACK IS NOT NESSARY, IF BGUNIX HAD UP-TO-DATE STL LIBRARIES
            // THIS WOULD NOT BE A PROBLEM.
            if(end < start)
                throw invalid_argument("Crossed iterators");
                
            clear();
            resize(end-start);        
        }
        
        OBVector(const OBVector<T> &t) : vector<T>(t) { }
        ~OBVector() { }

        reference operator[] (size_type n) {
            if(n > 0 && n < ((end() - begin()) + 1)) {
                return(*(begin() + (n - 1)));
            }
            else {
                throw out_of_range("Out of bounds of array");
            }
        }
        
        const_reference operator[] (size_type n) const {
            if(n > 0 && n < ((end() - begin()) + 1)) {
                return(*(begin() + (n - 1)));
            }
            else {
                throw out_of_range("Out of bounds of array");
            }
        }

        iterator erase(iterator first, iterator last) {
            if(first > last)
                throw invalid_argument("Crossed iterators");
            
            return(vector<T>::erase(first,last));
        }

        iterator erase(iterator pos) {
            if((begin() > pos) || ((end()-1) < pos))
                throw out_of_range("Iterator out of range");
                
            return(vector<T>::erase(pos));
        }

        iterator insert(iterator pos, const T &x) {
            if((begin() > pos) || ((end()-1) < pos))
                throw out_of_range("Iterator out of range");
                
            return(vector<T>::insert(pos, x));
        }

        iterator insert(iterator pos) {
            if((begin() > pos) || ((end()-1) < pos))
                throw out_of_range("Iterator out of range");
            return(insert(pos,T()));
        }

        void insert(iterator pos, const_iterator first, const_iterator last) {
            if(first > last)
                throw invalid_argument("Crossed iterators");
            if((begin() > pos) || ((end()-1) < pos))
                throw out_of_range("Iterator out of range");
                
            vector<T>::insert(pos,first,last);
        }
        
        void insert(iterator pos, size_type n, const T &x) {
            if((begin() > pos) || ((end()-1) < pos))
                throw out_of_range("Iterator out of range");
            if(((signed int)n) < 0)
                throw invalid_argument("Invalid size");
            
            vector<T>::insert(pos,n,x);
        }

        void insert(iterator pos, int n, const T &x) {
            if((begin() > pos) || ((end()-1) < pos))
                throw out_of_range("Iterator out of range");
            if(((signed int)n) < 0)
                throw invalid_argument("Invalid size");
            
            vector<T>::insert(pos,n,x);
        }
        
        void insert(iterator pos, long n, const T &x) {
            if((begin() > pos) || ((end()-1) < pos))
                throw out_of_range("Iterator out of range");
            if(((signed int)n) < 0)
                throw invalid_argument("Invalid size");
            
            vector<T>::insert(pos,n,x);
        }
        
    protected:
    private:
    
};

// FUNCTIONS ************************************************************

// **********************************************************************

#endif
