//  FNPTR.CPP
//  This sample program shows the use of an array of function pointers
//  and a pointer that points to member functions of a class.
//  The only public functions in the class are the constructor and a
//  small test program.

#include <iostream>
using std::cout;
using std::endl;

class Sample {

public:
   Sample();
   void test();

private:
   // Declare a function pointer variable and a function pointer array.
   // Without "Sample::" this array would contain pointers to C-style
   // global functions.
   void (Sample::*fnptr)();
   void (Sample::*fnptr_arr[3])();

   void a() { cout << "function a" << endl; }
   void b() { cout << "function b" << endl; }
   void c() { cout << "function c" << endl; }

};

Sample::Sample()
{
   fnptr = NULL;
   fnptr_arr[0] = &Sample::a;
   fnptr_arr[1] = &Sample::b;
   fnptr_arr[2] = &Sample::c;
}

void Sample::test()
{
   // Use the function pointer array to call private functions.
   // Note the use of the ->* operator to signify that we are calling
   // a function associated with the current object (this).
   (this->*fnptr_arr[0])();
   (this->*fnptr_arr[2])();

   // Use the function pointer variable to call private functions.
   fnptr = &Sample::a;
   (this->*fnptr)();
   fnptr = &Sample::b;
   (this->*fnptr)();
}

int main()
{
   Sample obj;
   obj.test();
   return 0;
}

