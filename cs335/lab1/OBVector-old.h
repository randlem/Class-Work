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
 *  iterator begin()
 *   Get an iterator to the first element.
 * 
 *  const_iterator begin()
 *   Get a constant iterator to the first element.
 *
 *  iterator end()
 *   Get an iterator to the last element.
 *
 *  const_iterator end()
 *   Get a constant iterator to the last elemnt.
 * 
 *  reverse_iterator rbegin()
 *   Get a reverse iterator to the first element.
 *  
 *  const_reverse_iterator rbegin()
 *   Get a constant reverse iterator to the first elemnt.
 *
 *  reverse_iterator rend()
 *   Get a reverse iterator to the last element.
 *
 *  const_reverse_iterator rend()
 *   Get a constant reverse iterator to the last element.
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

// NAMESPACES ***********************************************************
using std::vector;

// MACROS ***************************************************************

// CLASSES **************************************************************
template <typename T>
class OBVector : public vector<T> {
    public:
/*        typedef T value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type* iterator;
        typedef const value_type* const_iterator;
        typedef value_type& reference;
        typedef const value_type& const_reference;
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;

#ifdef __STL_CLASS_PARTIAL_SPECIALIZATION
        typedef reverse_iterator<const_iterator> const_reverse_iterator;
        typedef reverse_iterator<iterator> reverse_iterator;
#else /* __STL_CLASS_PARTIAL_SPECIALIZATION */
/*        typedef reverse_iterator<const_iterator, value_type, const_reference, difference_type>  const_reverse_iterator;
        typedef reverse_iterator<iterator, value_type, reference, difference_type> reverse_iterator;
#endif /* __STL_CLASS_PARTIAL_SPECIALIZATION */
    
        OBVector() : vector<T>() { }

        OBVector(size_type n) : vector<T>(n) { }

        OBVector(size_type n, const T& t) : vector<T>(n,t) { }

        OBVector(int n, const T& t) : vector<T>(n,t) { }

        OBVector(long n, const T& t) : vector<T>(n,t) { }

        OBVector(const_iterator start, const_iterator end) : vector<T>(start,end) { }

        OBVector(const OBVector<T> &t) : vector<T>(t) { }

        ~OBVector() { }

        iterator begin() { return(vector<T>::begin()+1); }

        const_iterator begin() const { return(vector<T>::begin()+1); }

        iterator end() { return(vector<T>::end()+1); }

        const_iterator end() const { return(vector<T>::end()+1); }
#ifdef __STL_CLASS_PARTIAL_SPECIALIZATION        
        reverse_iterator rbegin() { return(reverse_iterator(end())); }

        const_reverse_iterator rbegin() const { return(const_reverse_iterator(end())); }

        reverse_iterator rend() { return(reverse_iterator(begin())); }

        const_reverse_iterator rend() const { return(const_reverse_iterator(begin())); }
#endif /*__STL_CLASS_PARTIAL_SPECIALIAZATION */
        reference operator[] (size_type n) { return(*(begin() + n - 1)); }

        const_reference operator[] (size_type n) const { return(*(begin() + n - 1)); }

        iterator erase(iterator first, iterator last) { return(vector<T>::erase(first - 1,last - 1)); }

        iterator erase(iterator position) { return(vector<T>::erase(position - 1)); }

        iterator insert(iterator position, const T &x) { return(vector<T>::insert(position - 1, x)); }

        iterator insert(iterator position) { return(insert(position,T())); }

/*        void insert(iterator position, const_iterator first,const_iterator last);
        
        void insert(iterator pos, size_type n, const T &x);

        void insert(iterator pos, int n, const T &x);

        void insert(iterator pos, long n, const T &x);
*/
    protected:
    private:
    
};

// FUNCTIONS ************************************************************

// **********************************************************************

#endif
