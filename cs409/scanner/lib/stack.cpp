#include <iostream>
using std::cout;
using std::endl;

#include <stack>
using std::stack;

int main()
{
   stack<int> st;
   st.push(20);
   st.push(200);
   st.push(2002);

   cout << "There are " << st.size() << " stack entries."
        << endl << endl << "Stack contents:" << endl;

   while(!st.empty()) {
      cout << st.top() << endl;
      st.pop();
   }

   return 0;
}

