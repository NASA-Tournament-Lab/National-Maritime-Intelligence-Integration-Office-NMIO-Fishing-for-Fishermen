#include <bits/stdc++.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>

using namespace std;

double getTime() {
    timeval t;
    gettimeofday(&t,NULL);
    return 1e-6*t.tv_usec + t.tv_sec;
}
string trim(const string& str,const string& whitespace=" \t\r\n"){
    size_t strBegin = str.find_first_not_of(whitespace);
    if (strBegin == string::npos) return "";
    size_t strEnd = str.find_last_not_of(whitespace);
    size_t strRange = strEnd - strBegin + 1;
    return str.substr(strBegin, strRange);
}
vector <string> splitBy(const string& text, char by){   // split string by by
	vector <string> vys;
	stringstream ss(text);
    string word;
    while(getline(ss,word,by)){
        vys.push_back(word);
    }
    return vys;
}
vector <string> listdir(const string& folder){
	DIR *dir;
	struct dirent *ent;
	vector <string> files;
	if((dir = opendir(folder.c_str())) != NULL){
		while((ent = readdir(dir)) != NULL){
			string temp = ent->d_name;
			if(temp != "." && temp != "..") files.push_back(temp);
		}
		closedir(dir);
	} else{
		cerr << "Could not open directory [" << folder << "]" << endl;
	}
	sort(files.begin(), files.end());
	return files;
}

int main(int argc, char* argv[]){
	double start = getTime();
		
	vector <string> files = listdir("VesselTracks");
	ofstream out("lengths.csv");

	for(int i = 0; i < files.size(); i++){
		if(i % 10 == 0) clog << i << "\r";
		ifstream in(("VesselTracks/" + files[i]).c_str());
		vector <double> data;
		bool firstLine = true;
		while(!in.eof()){
			string line;
			getline(in, line);
			if(firstLine){
				firstLine = false;
				continue;
			}
			vector <string> row = splitBy(trim(line), ',');
			if(row.size() == 13){
				for(int i = 0; i < 12; i++){
					data.push_back(atof(row[i + 1].c_str()));
				}
			}
		}
		int length = data.size() / 12;
		string id = files[i];
		id.replace(id.end() - 4, id.end(), "");
		out << id << "," << length << endl;
		in.close();
		vector <double> datat(data.size());
		for(int j = 0; j < 12; j++) for(int k = 0; k < length; k++) datat[j * length + k] = data[k * 12 + j];
		ofstream outb(("VesselTracksBin/" + id + ".dat"), ios::binary);
		outb.write((const char *)&datat[0], datat.size() * sizeof(double));
		outb.close();
	}
	out.close();
	
	return 0;
}

