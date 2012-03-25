#ifndef __OPTIONS_H__
#define __OPTIONS_H__

#include <iostream>
using std::cout;
using std::endl;

#include <map>
using std::map;

#include <string>
using std::string;

class Options {
	public:
		Options() { }
		~Options() { }

		void setFlags(string flag, string option) {
			flags[flag.c_str()] = option;
			options[option] = "";
		}

		bool parseCmdLine(int argc, char* argv[]) {
			map<string, string>::iterator i;
			string s;

			if (argc < 1)
				return false;

			for(int z=1; z < argc; ++z) {
				s = argv[z];
				s.erase(0,1);

				i = flags.find(s);
				if (i != flags.end()) {
					string option = flags[(*i).first];
					options[option] = argv[++z];
				}
			}

			return true;
		}

		string getOption(string option) {
			return options[option];
		}

		void setOption(string option, string value) {
			options[option] = value;
		}

	protected:
		map<string, string> options;
		map<string, string> flags;
};

#endif
