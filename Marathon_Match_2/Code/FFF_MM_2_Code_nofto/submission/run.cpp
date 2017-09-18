#include <bits/stdc++.h>
#include <sys/time.h>
#define SSTR( x ) dynamic_cast< std::ostringstream & >( ( std::ostringstream() << std::dec << x ) ).str()
#define GCOST(x) ( sqrt((x)*(1 - (x)) ) )
#define SQ(x) ((x)*(x))
#define sind(x) (sin(fmod((x),360)*PI/180))
#define cosd(x) (cos(fmod((x),360)*PI/180))

using namespace std;

const int VAR2D = 3;
vector <int> var1 = {3, 3, 3};
vector <int> var2 = {4, 1, 2};
const int FEATURES = 1 + 2 * 12 + 6 + 6 + 2 + 9 + 5 + 2 * 5 + VAR2D * 5;
int FEATURETRY = 6;
const int CLASSES = 5;
vector <int> skipFeature(FEATURES, 0);
vector <int> toBeSkipped = {29, 23, 24, 22, 1, 2, 19, 39, 21, 32, 0, 7, 18, 36, 26, 16, 34, 12, 4, 33, 28, 20, 6, 14, 27, 38, 11, 30};//23, 24, 2, 1, 31, 35, 22, 33, 34, 21, 19, 7, 32, 0, 18, 16, 30, 26, 12, 4};

const int REG = 10;
const double regSize[REG] = {0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 60};//{0.05, 0.5, 5, 30};
vector <float> regStat[REG];
const double MINSHIP = 10.0;
const bool DISCARD = true;

vector <int> hist = {3};//3};//, 4};//, 5};//, 6};//, 7, 8, 9/*, 10, 11*/};
vector <int> hist2 = {3};//3};//, 4};//, 5};//, 6};//, 7, 8, 9/*, 10, 11*/};
vector <double> vmin = {      0.0, -90.0, -180.0,  0.2, -1600.0,  0.0, 30.0, -2.0, -5.0,    0.0, -2.0, -1.0}; 
vector <double> vmax = {8000000.0,  90.0,  180.0, 20.0,   100.0, 20.0, 40.0,  1.0, 35.0, 1000.0,  2.0,  1.0}; 
vector <double> vmin2 = {      0.0, -90.0, -180.0,  5.0, -1600.0,  0.0, 30.0, -2.0, -5.0,    0.0, -2.0, -1.0}; 
vector <double> vmax2 = {8000000.0,  90.0,  180.0, 100.0,   100.0, 20.0, 40.0,  1.0, 35.0, 1000.0,  2.0,  1.0}; 

const int HIST = 7;//10;
const int histSize[HIST] = {/*1024, 512, 256, */128, 64, 32, 16, 8, 4, 2};
vector <float> histStat[12][HIST];
vector <float> histStat2[12][HIST];
const double MINSHIP1D = 10.0;

const int HIST2D = 10;//10;
const int histSize2d[HIST2D] = {/*4096, 2048,*/ 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2};
vector <float> histStat2d[VAR2D][HIST2D];

const int CLUSTERS = 2;
const int MAXLEVEL = 50;
int MINNODEMSE = 5;
int MINNODE = 1;
int TREES = 1000;
const int MAXSAMPLESIZE = 3000000; //10000000;
int BORDER = 10000000;
const double EPS = 1e-10;

const int TRAINFILES = 1209;
const int TESTFILES = 1211;
const int SAMPLES = TRAINFILES;
float FEAT[FEATURES * SAMPLES];
float RESULT[CLASSES][SAMPLES];
bool training;
const double PI=3.141592653589793;

unordered_map <string, int> lengths;
unordered_map <	string, int> type = {{"trawler", 1}, {"seiner", 2}, {"longliner", 3}, {"support", 4}, {"other", 0}};
vector <string> types = {"other", "trawler", "seiner", "longliner", "support"};
const double NA[12] = {-99999, 91, 181, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999};

vector <int> featureScore;
vector <int> featureScoreC;

double getTime() {
    timeval t;
    gettimeofday(&t,NULL);
    return 1e-6*t.tv_usec + t.tv_sec;
}
// random generator
unsigned long long nowRand = 1;
void seedBig(unsigned long long seed){
	nowRand = seed;
}
unsigned long long randBig(){
	nowRand = ((nowRand * 6364136223846793005ULL + 1442695040888963407ULL) >> 1);
	return nowRand;
}
string int2len(int v,int l){
	string ret=SSTR(v);
	int dig=floor(log10(v+0.1))+1;
	while(ret.length()<l) ret=" "+ret;
	return ret;
}
string string2len(string ret, int l){
	while(ret.length()<l) ret+=" ";
	return ret;
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

vector <int> selectFeatures(){
	set <int> temp;
	while(temp.size() != FEATURETRY){
		int f = randBig() % FEATURES;
		if(skipFeature[f] == 0) temp.insert(f);
	}
	vector<int> result(temp.begin(), temp.end());
	return result;
}
int SORTBY = 0;
int SORTBYSAMPLES = 0;
class Node{
  public:
	int left;
	int right;
	int feature;
	float value;
	int level;
	int counts[CLUSTERS];
	int total;
	Node(){
		left = -1;
		right = -1;
		feature = -1;
		value = 0;
		level = -1;
		for(int i = 0; i < CLUSTERS; i++) counts[i] = 0;
		total = 0;
	}
	Node(int lev, const vector <int>& cou, int tot){
		left = -1;
		right = -1;
		feature = -1;
		value = 0;
		level = lev;
		copy(cou.begin(), cou.end(), counts);
		total = tot;
	}
};
class Tree{
  public:
	vector <Node> node;
	double ratioAtNode(int nodeId, float* source) const{
		if(node[nodeId].left == -1){
			double sum = 0;
			for(int cluster = 0; cluster < CLUSTERS; cluster++){
				sum += node[nodeId].counts[cluster] * cluster;
			}
			return sum / node[nodeId].total;
		}
		if(*(source + node[nodeId].feature) <= node[nodeId].value) return ratioAtNode(node[nodeId].left, source);
		return ratioAtNode(node[nodeId].right, source);
	}
	double assignRatio(float* source) const{
		return ratioAtNode(0, source);
	}
	void divideNode(int nodeIndex, vector <int>& sample, int clas){
		int n = sample.size();
		int nonzero = 0;
		for(int i = 0; i < CLUSTERS; i++){
			if(node[nodeIndex].counts[i] > 0) nonzero++;
			if(nonzero > 1) break;
		}
		if(node[nodeIndex].level < MAXLEVEL - 1 && nonzero > 1 && node[nodeIndex].total > MINNODE){
	    	vector <int> feaID = selectFeatures();
	    	double minCost = 1e30;
	    	int bestF = -1;
	    	double bestValue = 0;
	    	int bestI = 0;
	    	vector <int> bestC1(CLUSTERS, 0);
	    	int bestTotalL = 0;
			for(int f = 0; f < FEATURETRY; f++){
				SORTBY = feaID[f];
				SORTBYSAMPLES = SORTBY * SAMPLES;
				sort(sample.begin(), sample.end(), [&](int aa, int bb){return FEAT[SORTBYSAMPLES + aa] < FEAT[SORTBYSAMPLES + bb];});
				vector <int> c1(CLUSTERS, 0);
	    		int totalL = 0;
	    		for(int i = 0; i < n-1; i++){
	    			c1[int(RESULT[clas][sample[i]])]++;
	    			totalL++;
	    			if(FEAT[SORTBYSAMPLES + sample[i+1]] > FEAT[SORTBYSAMPLES + sample[i]]){
	    			    double costL = 0.0;
						double costR = 0.0;
						for(int cl = 0; cl < CLUSTERS; cl++){
							costL += GCOST(c1[cl] / static_cast<double>(totalL));
							costR += GCOST((node[nodeIndex].counts[cl] - c1[cl]) / static_cast<double>(n - totalL));
						}
						double cost = (totalL * costL + (n - totalL) * costR) / n;
						if(cost < minCost && i >= n/BORDER && i < n - n/BORDER){
	    			    	minCost = cost;
	    			    	bestF = feaID[f];
	    			    	bestValue = FEAT[SORTBYSAMPLES + sample[i]];
							bestI = i;
	    			    	bestC1 = c1;
	    			    	bestTotalL = totalL;
	    			    }
					}
	    		}
	    	}
	    	if(bestF >= 0){
				featureScore[bestF] += n;
				featureScoreC[bestF]++;
		    	vector <int> sampleLeft; sampleLeft.reserve(bestI + 1);
		    	vector <int> sampleRight; sampleRight.reserve(n - bestI - 1);
		    	SORTBYSAMPLES = bestF * SAMPLES;
				for(int i = 0; i < n; i++){
		    		if(FEAT[SORTBYSAMPLES + sample[i]] <= bestValue){
						sampleLeft.push_back(sample[i]);
					}
		    		else sampleRight.push_back(sample[i]);
		    	}
		        node[nodeIndex].feature = bestF;
		    	node[nodeIndex].value = bestValue;
		    	node.push_back(Node(node[nodeIndex].level + 1, bestC1, bestTotalL));
		    	node[nodeIndex].left = node.size() - 1;
		    	vector <int> c2(CLUSTERS, 0);
		    	for(int i = 0; i < CLUSTERS; i++){
		    		c2[i] = node[nodeIndex].counts[i] - bestC1[i];	
		    	}
		    	node.push_back(Node(node[nodeIndex].level + 1, c2, node[nodeIndex].total - bestTotalL));
		    	node[nodeIndex].right = node.size() - 1;
			    divideNode(node[nodeIndex].left, sampleLeft, clas);
				divideNode(node[nodeIndex].right, sampleRight, clas);
			}
		}
	}
	Tree(){
	}
	int toStream(ofstream &out){
		int size = node.size();
		out.write((const char *)&size, sizeof(int));
		for(int i = 0; i < size; i++){
			out.write((const char *)&node[i], sizeof(Node));
		}
		return 0;
	}
	Tree(ifstream& in){
		int size;
		in.read((char *)&size, sizeof(int));
		node.resize(size);
		for(int i = 0; i < size; i++){
			in.read((char *)&node[i], sizeof(Node));
		}
	}
};
int RFtoFile(vector <Tree>& rf, string fileName){
	ofstream out(fileName.c_str(), ios::binary);
	int trees = rf.size();
	out.write((const char *)&trees, sizeof(int));
	for(int i = 0; i < trees; i++){
		rf[i].toStream(out);
	}
	out.close();
	return 0;
}
vector <Tree> RFfromFile(string fileName){
	vector <Tree> rf; 
	ifstream in(fileName.c_str(), ios::binary);
	int trees;
	in.read((char *)&trees, sizeof(int));
	for(int i = 0; i < trees; i++){
		rf.push_back(Tree(in));
	}
	in.close();
	return rf;
}
double forestAssignResult(const vector <Tree>& tree, float* source){
	double result = 0;
	for(int t = 0; t < tree.size(); t++){
		result += tree[t].assignRatio(source);
	}
	return result / tree.size();
}
Tree buildTree(int n, int clas){
	int ns = min(n, MAXSAMPLESIZE);
	vector <int> sample;
	sample.resize(ns);
	Tree tree;
	tree.node.resize(1, Node());
	tree.node[0].level = 0;
	for(int i = 0; i < ns; i++){
		sample[i] = randBig() % n;
		tree.node[0].counts[int(RESULT[clas][sample[i]])]++;
	}
	tree.node[0].total = ns;
	tree.divideNode(0, sample, clas);
	return tree;
}

struct Point3D{
	double x;
	double y;
	double z;
	Point3D(){}
	Point3D(double xx, double yy, double zz) : x(xx), y(yy), z(zz){}
	Point3D(double lat, double lon){
		if(lat == 91 || lon == 181){
			x = -99999; y = -99999; z = -99999;
		}
		else{
			x = 6371.01 * cosd(lat) * cosd(lon);
			y = 6371.01 * cosd(lat) * sind(lon);
			z = 6371.01 * sind(lat);			
		}
	}
};
inline bool operator==(const Point3D& lhs, const Point3D& rhs){	
    return lhs.x==rhs.x && lhs.y==rhs.y && lhs.z==rhs.z;
}

inline double angleF(Point3D v, Point3D p1, Point3D p2){
	if(v == p1 || v == p2) return 180.0;
	Point3D v1 = Point3D(p1.x - v.x, p1.y - v.y, p1.z - v.z);
	Point3D v2 = Point3D(p2.x - v.x, p2.y - v.y, p2.z - v.z);
	return acos((v1.x*v2.x + v1.y*v2.y + v1.z*v2.z) / sqrt((v1.x*v1.x + v1.y*v1.y + v1.z*v1.z) * (v2.x*v2.x + v2.y*v2.y + v2.z*v2.z))) / 3.14159265358979 * 180;
}

double earthDistance(double loc1lat, double loc1lon, double loc2lat, double loc2lon){
	double deltaLon = fabs(loc1lon - loc2lon);
	if(deltaLon > 180) deltaLon = 360 - deltaLon;
	return 6371.01 * atan2(sqrt(SQ(cosd(loc1lat) * sind(deltaLon)) + SQ(cosd(loc2lat)*sind(loc1lat) - sind(loc2lat)*cosd(loc1lat)*cosd(deltaLon))), sind(loc2lat)*sind(loc1lat) + cosd(loc2lat)*cosd(loc1lat)*cosd(deltaLon));
}

vector <float> stats(double* source, int L, double na){
	double s = 0;
	double ss = 0;
	double sss = 0;
	int size = 0;
	if(L == 0) return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	double prevv = *source;
	double nextv = *source;
	double minv = *source;
	double maxv = * source;
	for(int i = 0; i < L; i++){
		double v = *(source + i);
		if(na != -99999){
			double dif0 = i == 0 ? fabs(v - *(source + i + 1)) : fabs(v - *(source + i - 1));
			double dif1 = i == L - 1 ? fabs(v - *(source + i - 1)) : fabs(v - *(source + i + 1));
			if(dif0 > 60 && dif0 < 300 && dif1 > 60 && dif1 < 300) continue;
		}
		if(na == 181){
			while(v - prevv > 270) v -= 360;
			while(v - prevv < -270) v += 360;
			prevv = v;
		}
		if(v != na){
			minv = min(minv, v);
			maxv = max(maxv, v);
			s += v;
			ss += v * v;
			sss += v * v * v;
			size++;
		}
	}
	if(size == 0) size = 1;
	double avg = s / size;
	double var = (ss - s * s / size) / size;
	double temp = (sss - 3 * ss * s / size + 2 * s * s * s / size / size) / size;
	double skew = var == 0.0 ? 0.0 : temp / sqrt(var * var * var);
	float avgf = avg;
	float varf = var;
	float skewf = skew;
	if(isnan(avgf)) avgf = 0;
	if(isnan(varf)) varf = 0;
	if(isnan(skewf)) skewf = 0;
	return {avgf, varf, skewf, float(minv), float(maxv), float(maxv - minv)};
}

vector <float> dstats(double* time, double* lat, double* lon, int L){
	vector <double> seg;
	vector <double> dif;
	vector <double> speed;
	for(int i = 0; i < L - 1; i++){
		double lat0 = *(lat + i);
		double lon0 = *(lon + i);
		double lat1 = *(lat + i + 1);
		double lon1 = *(lon + i + 1);
		if(lat0 != 91 && lon0 != 181 && lat1 != 91 && lon1 != 181){
			double tdif = *(time + i + 1) - *(time + i);
			double dis = earthDistance(lat0, lon0, lat1, lon1);
			double sp = dis / tdif * 3600;
			if(sp < 100){
				seg.push_back(dis);
				dif.push_back(tdif);
				speed.push_back(sp);
			}
		}
	}
	int LL = seg.size();
	vector <float> segf = stats(&seg[0], LL, -99999);
	vector <float> diff = stats(&dif[0], LL, -99999);
	vector <float> speedf = stats(&speed[0], LL, -99999);
	return {segf[0], segf[1], diff[0], diff[1], speedf[0], speedf[1]};
}

vector <float> histStats2d(double value1, double value2, int f, int h, double shipsIn){
	int f1 = var1[f];
	int f2 = var2[f];
	if(h == HIST2D || value1 < vmin[f1] || value1 > vmax[f1] || value2 < vmin[f2] || value2 > vmax[f2]) return {230075.0f, 3668175.0f, 1576067.0f, 1353528.0f, 1241188.0f};
	vector <float> ans(5, 0);
	double rs1 = (vmax[f1] - vmin[f1]) / histSize2d[h];
	double rs2 = (vmax[f2] - vmin[f2]) / histSize2d[h];
	int W = histSize2d[h];
	double xf = (value1 - vmin[f1]) / rs1;
	double yf = (value2 - vmin[f2]) / rs2;
	int x = floor(xf);
	int y = floor(yf);
	double px = xf - x;
	double py = yf - y;
	double ships = 0;
	if(x < W + 1 && y < W + 1){
		for(int t = 0; t < 5; t++) ans[t] += histStat2d[f][h][6 * (y * (W + 1) + x) + t] * (1 - px) * (1 - py);
		ships += histStat2d[f][h][6 * (y * (W + 1) + x) + 5] * (1 - px) * (1 - py);
		if(x + 1 < W + 1){
			for(int t = 0; t < 5; t++) ans[t] += histStat2d[f][h][6 * (y * (W + 1) + x + 1) + t] * px * (1 - py);
			ships += histStat2d[f][h][6 * (y * (W + 1) + x + 1) + 5] * px * (1 - py);
		}
		if(y + 1 < W + 1){
			for(int t = 0; t < 5; t++) ans[t] += histStat2d[f][h][6 * ((y + 1) * (W + 1) + x) + t] * (1 - px) * py;
			ships += histStat2d[f][h][6 * ((y + 1) * (W + 1) + x) + 5] * (1 - px) * py;
			if(x + 1 < W + 1){
				for(int t = 0; t < 5; t++) ans[t] += histStat2d[f][h][6 * ((y + 1) * (W + 1) + x + 1) + t] * px * py;
				ships += histStat2d[f][h][6 * ((y + 1) * (W + 1) + x + 1) + 5] * px * py;
			}
		}
	}
	if(ships >= shipsIn) return ans;
	vector <float> ansUp = histStats2d(value1, value2, f, h + 1, shipsIn);
	for(int t = 0; t < 5; t++) ans[t] = (ans[t] * ships + ansUp[t] * (shipsIn - ships)) / shipsIn;
	return ans;
}
vector <float> histStats(double value, int f, int h, double shipsIn){
	if(h == HIST || value < vmin[f] || value > vmax[f]) return {230075.0f, 3668175.0f, 1576067.0f, 1353528.0f, 1241188.0f};
	vector <float> ans(5, 0);
	double rs = (vmax[f] - vmin[f]) / histSize[h];
	int W = histSize[h];
	double xf = (value - vmin[f]) / rs;
	int x = floor(xf);
	double px = xf - x;
	double ships = 0;
	if(x < W + 1){
		for(int t = 0; t < 5; t++) ans[t] += histStat[f][h][6 * x + t] * (1 - px);
		ships += histStat[f][h][6 * x + 5] * (1 - px);
		if(x + 1 < W + 1){
			for(int t = 0; t < 5; t++) ans[t] += histStat[f][h][6 * (x + 1) + t] * px;
			ships += histStat[f][h][6 * (x + 1) + 5] * px;
		}
	}
	if(ships >= shipsIn) return ans;
	vector <float> ansUp = histStats(value, f, h + 1, shipsIn);
	for(int t = 0; t < 5; t++) ans[t] = (ans[t] * ships + ansUp[t] * (shipsIn - ships)) / shipsIn;
	return ans;	
}
vector <float> histStats2(double value, int f, int h, double shipsIn){
	if(h == HIST || value < vmin2[f] || value > vmax2[f]) return {230075.0f, 3668175.0f, 1576067.0f, 1353528.0f, 1241188.0f};
	vector <float> ans(5, 0);
	double rs = (vmax2[f] - vmin2[f]) / histSize[h];
	int W = histSize[h];
	double xf = (value - vmin2[f]) / rs;
	int x = floor(xf);
	double px = xf - x;
	double ships = 0;
	if(x < W + 1){
		for(int t = 0; t < 5; t++) ans[t] += histStat2[f][h][6 * x + t] * (1 - px);
		ships += histStat2[f][h][6 * x + 5] * (1 - px);
		if(x + 1 < W + 1){
			for(int t = 0; t < 5; t++) ans[t] += histStat2[f][h][6 * (x + 1) + t] * px;
			ships += histStat2[f][h][6 * (x + 1) + 5] * px;
		}
	}
	if(ships >= shipsIn) return ans;
	vector <float> ansUp = histStats2(value, f, h + 1, shipsIn);
	for(int t = 0; t < 5; t++) ans[t] = (ans[t] * ships + ansUp[t] * (shipsIn - ships)) / shipsIn;
	return ans;	
}

vector <float> regionStats(double lat, double lon, int r, double shipsIn){
	if(r == REG || lat < -90 || lat > 90 || lon < -180 || lon > 180) return {230075.0f, 3668175.0f, 1576067.0f, 1353528.0f, 1241188.0f};
	vector <float> ans(5, 0);
	double rs = regSize[r];
	int W = round(360 / rs);
	int H = round(180 / rs);
	double xf = (lon + 180) / rs;
	double yf = (lat + 90) / rs;
	int x = floor(xf);
	int y = floor(yf);
	double px =	xf - x;
	double py = yf - y;
	double ships = 0;
	if(y < H + 1){
		for(int t = 0; t < 5; t++) ans[t] += regStat[r][6 * (y * W + x % W) + t] * (1 - px) * (1 - py);
		ships += regStat[r][6 * (y * W + x % W) + 5] * (1 - px) * (1 - py);
		for(int t = 0; t < 5; t++) ans[t] += regStat[r][6 * (y * W + (x + 1) % W) + t] * px * (1 - py);
		ships += regStat[r][6 * (y * W + (x + 1) % W) + 5] * px * (1 - py);
		if(y + 1 < H + 1){
			for(int t = 0; t < 5; t++) ans[t] += regStat[r][6 * ((y + 1) * W + x % W) + t] * (1 - px) * py;
			ships += regStat[r][6 * ((y + 1) * W + x % W) + 5] * (1 - px) * py;
			for(int t = 0; t < 5; t++) ans[t] += regStat[r][6 * ((y + 1) * W + (x + 1) % W) + t] * px * py;
			ships += regStat[r][6 * ((y + 1) * W + (x + 1) % W) + 5] * px * py;
		}
	}
	if(ships >= shipsIn) return ans;
	vector <float> ansUp = regionStats(lat, lon, r + 1, shipsIn);
	for(int t = 0; t < 5; t++) ans[t] = (ans[t] * ships + ansUp[t] * (shipsIn - ships)) / shipsIn;
	return ans;
}

int processShipForRegionStats(string id, int t){
	int L = lengths[id];
	vector <double> data(L * 12);
	ifstream inw(("VesselTracksBin/" + id + ".dat"), ios::binary);
	inw.read((char *)&data[0], L * 12 * sizeof(double));
	inw.close();
	
	for(int f = 0; f < VAR2D; f++){
		int f1 = var1[f];
		int f2 = var2[f];
		for(int h = 0; h < HIST2D; h++){
			int W = histSize2d[h];
			double rs1 = (vmax[f1] - vmin[f1]) / W;
			double rs2 = (vmax[f2] - vmin[f2]) / W;
			vector <bool> was((W + 1) * (W + 1), false);
			for(int i = 0; i < L; i++){
				double val1 = data[f1 * L + i];
				double val2 = data[f2 * L + i];
				if(val1 >= vmin[f1] && val1 <= vmax[f1] && val2 >= vmin[f2] && val2 <= vmax[f2]){
					double xf = (val1 - vmin[f1]) / rs1;
					double yf = (val2 - vmin[f2]) / rs2;
					int x = floor(xf);
					int y = floor(yf);
					double px = xf - x;
					double py = yf - y;
					if(x < W + 1 && y < W + 1){
						histStat2d[f][h][6 * (y * (W + 1) + x) + t] += (1 - px) * (1 - py);
						if(!was[y * (W + 1) + x]){
							was[y * (W + 1) + x] = true;
							histStat2d[f][h][6 * (y * (W + 1) + x) + 5] += 1;
						}
						if(x + 1 < W + 1){
							histStat2d[f][h][6 * (y * (W + 1) + x + 1) + t] += px * (1 - py);
							if(!was[y * (W + 1) + x + 1]){
								was[y * (W + 1) + x + 1] = true;
								histStat2d[f][h][6 * (y * (W + 1) + x + 1) + 5] += 1;
							}
						}
						if(y + 1 < W + 1){
							histStat2d[f][h][6 * ((y + 1) * (W + 1) + x) + t] += (1 - px) * py;
							if(!was[(y + 1) * (W + 1) + x]){
								was[(y + 1) * (W + 1) + x] = true;
								histStat2d[f][h][6 * ((y + 1) * (W + 1) + x) + 5] += 1;
							}
							if(x + 1 < W + 1){
								histStat2d[f][h][6 * ((y + 1) * (W + 1) + x + 1) + t] += px * py;
								if(!was[(y + 1) * (W + 1) + x + 1]){
									was[(y + 1) * (W + 1) + x + 1] = true;
									histStat2d[f][h][6 * ((y + 1) * (W + 1) + x + 1) + 5] += 1;
								}
							}							
						}
					}
				}
			}
		}
	}
	for(int f : hist){
		for(int h = 0; h < HIST; h++){
			int W = histSize[h];
			double rs = (vmax[f] - vmin[f]) / W;
			vector <bool> was(W + 1, false);
			for(int i = 0; i < L; i++){
				double val = data[f * L + i];
				if(val >= vmin[f] && val <= vmax[f]){
					double xf = (val - vmin[f]) / rs;
					int x = floor(xf);
					double px = xf - x;
					if(x < W + 1){
						histStat[f][h][6 * x + t] += 1 - px;
						if(!was[x]){
							was[x] = true;
							histStat[f][h][6 * x + 5] += 1;
						}
						if(x + 1 < W + 1){
							histStat[f][h][6 * (x + 1) + t] += px;
							if(!was[x + 1]){
								was[x + 1] = true;
								histStat[f][h][6 * (x + 1) + 5] += 1;
							}
						}
					}
				}
			}
		}
	}
	for(int f : hist2){
		for(int h = 0; h < HIST; h++){
			int W = histSize[h];
			double rs = (vmax2[f] - vmin2[f]) / W;
			vector <bool> was(W + 1, false);
			for(int i = 0; i < L; i++){
				double val = data[f * L + i];
				if(val >= vmin2[f] && val <= vmax2[f]){
					double xf = (val - vmin2[f]) / rs;
					int x = floor(xf);
					double px = xf - x;
					if(x < W + 1){
						histStat2[f][h][6 * x + t] += 1 - px;
						if(!was[x]){
							was[x] = true;
							histStat2[f][h][6 * x + 5] += 1;
						}
						if(x + 1 < W + 1){
							histStat2[f][h][6 * (x + 1) + t] += px;
							if(!was[x + 1]){
								was[x + 1] = true;
								histStat2[f][h][6 * (x + 1) + 5] += 1;
							}
						}
					}
				}
			}
		}
	}
	
	for(int r = 0; r < REG; r++){
		double rs = regSize[r];
		int W = round(360 / rs);
		int H = round(180 / rs);
		vector <bool> was(W * (H + 1), false);
		for(int i = 0; i < L; i++){
			double lat = data[L + i];
			double lon = data[2 * L + i];
			if(lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180){
				double xf = (lon + 180) / rs;
				double yf = (lat + 90) / rs;
				int x = floor(xf);
				int y = floor(yf);
				double px =	xf - x;
				double py = yf - y;
				if(y < H + 1){
					regStat[r][6 * (y * W + x % W) + t] += (1 - px) * (1 - py);
					if(!was[y * W + x % W]){
						was[y * W + x % W] = true;
						regStat[r][6 * (y * W + x % W) + 5] += 1;
					}
					regStat[r][6 * (y * W + (x + 1) % W) + t] += px * (1 - py);
					if(!was[y * W + (x + 1) % W]){
						was[y * W + (x + 1) % W] = true;
						regStat[r][6 * (y * W + (x + 1) % W) + 5] += 1;
					}
					if(y + 1 < H + 1){
						regStat[r][6 * ((y + 1) * W + x % W) + t] += (1 - px) * py;
						if(!was[(y + 1) * W + x % W]){
							was[(y + 1) * W + x % W] = true;
							regStat[r][6 * ((y + 1) * W + x % W) + 5] += 1;
						}
						regStat[r][6 * ((y + 1) * W + (x + 1) % W) + t] += px * py;
						if(!was[(y + 1) * W + (x + 1) % W]){
							was[(y + 1) * W + (x + 1) % W] = true;
							regStat[r][6 * ((y + 1) * W + (x + 1) % W) + 5] += 1;
						}
					}
				}
			}
		}
	}
	return 0;
}
int processShipForRegionStatsBack(double* data, int L, int mult, int t){
	for(int f = 0; f < VAR2D; f++){
		int f1 = var1[f];
		int f2 = var2[f];
		for(int h = 0; h < HIST2D; h++){
			int W = histSize2d[h];
			double rs1 = (vmax[f1] - vmin[f1]) / W;
			double rs2 = (vmax[f2] - vmin[f2]) / W;
			vector <bool> was((W + 1) * (W + 1), false);
			for(int i = 0; i < L; i++){
				double val1 = data[f1 * L + i];
				double val2 = data[f2 * L + i];
				if(val1 >= vmin[f1] && val1 <= vmax[f1] && val2 >= vmin[f2] && val2 <= vmax[f2]){
					double xf = (val1 - vmin[f1]) / rs1;
					double yf = (val2 - vmin[f2]) / rs2;
					int x = floor(xf);
					int y = floor(yf);
					double px = xf - x;
					double py = yf - y;
					if(x < W + 1 && y < W + 1){
						histStat2d[f][h][6 * (y * (W + 1) + x) + t] += mult * (1 - px) * (1 - py);
						if(!was[y * (W + 1) + x]){
							was[y * (W + 1) + x] = true;
							histStat2d[f][h][6 * (y * (W + 1) + x) + 5] += mult;
						}
						if(x + 1 < W + 1){
							histStat2d[f][h][6 * (y * (W + 1) + x + 1) + t] += mult * px * (1 - py);
							if(!was[y * (W + 1) + x + 1]){
								was[y * (W + 1) + x + 1] = true;
								histStat2d[f][h][6 * (y * (W + 1) + x + 1) + 5] += mult;
							}
						}
						if(y + 1 < W + 1){
							histStat2d[f][h][6 * ((y + 1) * (W + 1) + x) + t] += mult * (1 - px) * py;
							if(!was[(y + 1) * (W + 1) + x]){
								was[(y + 1) * (W + 1) + x] = true;
								histStat2d[f][h][6 * ((y + 1) * (W + 1) + x) + 5] += mult;
							}
							if(x + 1 < W + 1){
								histStat2d[f][h][6 * ((y + 1) * (W + 1) + x + 1) + t] += mult * px * py;
								if(!was[(y + 1) * (W + 1) + x + 1]){
									was[(y + 1) * (W + 1) + x + 1] = true;
									histStat2d[f][h][6 * ((y + 1) * (W + 1) + x + 1) + 5] += mult;
								}
							}							
						}
					}
				}
			}
		}
	}
	
	for(int f : hist){
		for(int h = 0; h < HIST; h++){
			int W = histSize[h];
			double rs = (vmax[f] - vmin[f]) / W;
			vector <bool> was(W + 1, false);
			for(int i = 0; i < L; i++){
				double val = data[f * L + i];
				if(val >= vmin[f] && val <= vmax[f]){
					double xf = (val - vmin[f]) / rs;
					int x = floor(xf);
					double px = xf - x;
					if(x < W + 1){
						histStat[f][h][6 * x + t] += mult * (1 - px);
						if(!was[x]){
							was[x] = true;
							histStat[f][h][6 * x + 5] += mult;
						}
						if(x + 1 < W + 1){
							histStat[f][h][6 * (x + 1) + t] += mult * px;
							if(!was[x + 1]){
								was[x + 1] = true;
								histStat[f][h][6 * (x + 1) + 5] += mult;
							}
						}
					}
				}
			}
		}
	}
	for(int f : hist2){
		for(int h = 0; h < HIST; h++){
			int W = histSize[h];
			double rs = (vmax2[f] - vmin2[f]) / W;
			vector <bool> was(W + 1, false);
			for(int i = 0; i < L; i++){
				double val = data[f * L + i];
				if(val >= vmin2[f] && val <= vmax2[f]){
					double xf = (val - vmin2[f]) / rs;
					int x = floor(xf);
					double px = xf - x;
					if(x < W + 1){
						histStat2[f][h][6 * x + t] += mult * (1 - px);
						if(!was[x]){
							was[x] = true;
							histStat2[f][h][6 * x + 5] += mult;
						}
						if(x + 1 < W + 1){
							histStat2[f][h][6 * (x + 1) + t] += mult * px;
							if(!was[x + 1]){
								was[x + 1] = true;
								histStat2[f][h][6 * (x + 1) + 5] += mult;
							}
						}
					}
				}
			}
		}
	}
	
	for(int r = 0; r < REG; r++){
		double rs = regSize[r];
		int W = round(360 / rs);
		int H = round(180 / rs);
		vector <bool> was(W * (H + 1), false);
		for(int i = 0; i < L; i++){
			double lat = data[L + i];
			double lon = data[2 * L + i];
			if(lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180){
				double xf = (lon + 180) / rs;
				double yf = (lat + 90) / rs;
				int x = floor(xf);
				int y = floor(yf);
				double px =	xf - x;
				double py = yf - y;
				if(y < H + 1){
					regStat[r][6 * (y * W + x % W) + t] += mult * (1 - px) * (1 - py);
					if(!was[y * W + x % W]){
						was[y * W + x % W] = true;
						regStat[r][6 * (y * W + x % W) + 5]  += mult;
					}
					regStat[r][6 * (y * W + (x + 1) % W) + t] += mult * px * (1 - py);
					if(!was[y * W + (x + 1) % W]){
						was[y * W + (x + 1) % W] = true;
						regStat[r][6 * (y * W + (x + 1) % W) + 5] += mult;
					}
					if(y + 1 < H + 1){
						regStat[r][6 * ((y + 1) * W + x % W) + t] += mult * (1 - px) * py;
						if(!was[(y + 1) * W + x % W]){
							was[(y + 1) * W + x % W] = true;
							regStat[r][6 * ((y + 1) * W + x % W) + 5] += mult;
						}
						regStat[r][6 * ((y + 1) * W + (x + 1) % W) + t] += mult * px * py;
						if(!was[(y + 1) * W + (x + 1) % W]){
							was[(y + 1) * W + (x + 1) % W] = true;
							regStat[r][6 * ((y + 1) * W + (x + 1) % W) + 5] += mult;
						}
					}
				}
			}
		}
	}
	return 0;
}

int processShip(string id, float* target, bool discard, int t){
	int L = lengths[id];
	vector <double> data(L * 12);
	ifstream inw(("VesselTracksBin/" + id + ".dat"), ios::binary);
	inw.read((char *)&data[0], L * 12 * sizeof(double));
	inw.close();
	
	vector <double> x(L);
	vector <double> y(L);
	vector <double> z(L);
	vector <Point3D> p;
	for(int i = 0; i < L; i++){
		Point3D pi(data[L + i], data[2 * L + i]);
		x[i] = pi.x;
		y[i] = pi.y;
		z[i] = pi.z;
		if(pi.x != -99999) p.push_back(pi);
	}
	vector <double> angle;
	for(int i = 2; i < p.size(); i++){
		angle.push_back(angleF(p[i - 1], p[i-2], p[i]));
	}
	
	int shiftL = training ? SAMPLES : 1; 
	int shiftG = training ? 1 : FEATURES;
	float *targetL = target;
	*targetL = L; targetL += shiftL;
	for(int i = 0; i < 12; i++){
		vector <float> st = stats(&data[i * L], L, NA[i]);
		*targetL = st[0]; targetL += shiftL;
		*targetL = st[1]; targetL += shiftL;
	}

	vector <float> st;
	st = stats(&x[0], L, -99999);
	*targetL = st[0]; targetL += shiftL;
	*targetL = st[1]; targetL += shiftL;
	*targetL = st[3]; targetL += shiftL;
	*targetL = st[4]; targetL += shiftL;
	*targetL = st[5]; targetL += shiftL;
	st = stats(&y[0], L, -99999);
	*targetL = st[0]; targetL += shiftL;
	*targetL = st[1]; targetL += shiftL;
	*targetL = st[3]; targetL += shiftL;
	*targetL = st[4]; targetL += shiftL;
	*targetL = st[5]; targetL += shiftL;
	st = stats(&z[0], L, -99999);
	*targetL = st[0]; targetL += shiftL;
	*targetL = st[1]; targetL += shiftL;
	*targetL = st[3]; targetL += shiftL;
	*targetL = st[4]; targetL += shiftL;
	*targetL = st[5]; targetL += shiftL;

	st = dstats(&data[0], &data[L], &data[2 * L], L);	
	*targetL = st[0]; targetL += shiftL;
	*targetL = st[1]; targetL += shiftL;
	*targetL = st[2]; targetL += shiftL;
	*targetL = st[3]; targetL += shiftL;
	*targetL = st[4]; targetL += shiftL;
	*targetL = st[5]; targetL += shiftL;

	st = stats(&angle[0], angle.size(), -99999);
	*targetL = st[0]; targetL += shiftL;
	*targetL = st[1]; targetL += shiftL;

	if(discard) processShipForRegionStatsBack(&data[0], L, -1, t);
	vector <float> regs(5);
	int items = 0;
	for(int i = 0; i < L; i++){
		if(data[L + i] != 91 && data[2 * L + i] != 181){
			items++;
			vector <float> temp = regionStats(data[L + i], data[2 * L + i], 0, MINSHIP);
			float tempSum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
			regs[0] += temp[0] / tempSum;
			regs[1] += temp[1] / tempSum;
			regs[2] += temp[2] / tempSum;
			regs[3] += temp[3] / tempSum;
			regs[4] += temp[4] / tempSum;
		}
	}
	if(items == 0) items = 1;
	*targetL = regs[0] / items; targetL += shiftL;
	*targetL = regs[1] / items; targetL += shiftL;
	*targetL = regs[2] / items; targetL += shiftL;
	*targetL = regs[3] / items; targetL += shiftL;
	*targetL = regs[4] / items; targetL += shiftL;
	
	for(int f : hist){
		vector <float> his(5);
		int items = 0;
		for(int i = 0; i < L; i++){
			if(data[f * L + i] != NA[f]){
				items++;
				vector <float> temp = histStats(data[f * L + i], f, 0, MINSHIP1D);
				float tempSum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
				his[0] += temp[0] / tempSum;
				his[1] += temp[1] / tempSum;
				his[2] += temp[2] / tempSum;
				his[3] += temp[3] / tempSum;
				his[4] += temp[4] / tempSum;
			}
		}
		if(items == 0) items = 1;
		*targetL = his[0] / items; targetL += shiftL;
		*targetL = his[1] / items; targetL += shiftL;
		*targetL = his[2] / items; targetL += shiftL;
		*targetL = his[3] / items; targetL += shiftL;
		*targetL = his[4] / items; targetL += shiftL;	
	}
	for(int f : hist2){
		vector <float> his(5);
		int items = 0;
		for(int i = 0; i < L; i++){
			if(data[f * L + i] != NA[f]){
				items++;
				vector <float> temp = histStats2(data[f * L + i], f, 0, MINSHIP1D);
				float tempSum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
				his[0] += temp[0] / tempSum;
				his[1] += temp[1] / tempSum;
				his[2] += temp[2] / tempSum;
				his[3] += temp[3] / tempSum;
				his[4] += temp[4] / tempSum;
			}
		}
		if(items == 0) items = 1;
		*targetL = his[0] / items; targetL += shiftL;
		*targetL = his[1] / items; targetL += shiftL;
		*targetL = his[2] / items; targetL += shiftL;
		*targetL = his[3] / items; targetL += shiftL;
		*targetL = his[4] / items; targetL += shiftL;	
	}
	for(int f = 0; f < VAR2D; f++){
		int f1 = var1[f];
		int f2 = var2[f];
		vector <float> his(5);
		int items = 0;
		for(int i = 0; i < L; i++){
			if(data[f1 * L + i] != NA[f1] && data[f2 * L + i] != NA[f2]){
				items++;
				vector <float> temp = histStats2d(data[f1 * L + i], data[f2 * L + i], f, 0, MINSHIP);
				float tempSum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
				his[0] += temp[0] / tempSum;
				his[1] += temp[1] / tempSum;
				his[2] += temp[2] / tempSum;
				his[3] += temp[3] / tempSum;
				his[4] += temp[4] / tempSum;
			}
		}
		if(items == 0) items = 1;
		*targetL = his[0] / items; targetL += shiftL;
		*targetL = his[1] / items; targetL += shiftL;
		*targetL = his[2] / items; targetL += shiftL;
		*targetL = his[3] / items; targetL += shiftL;
		*targetL = his[4] / items; targetL += shiftL;	
	}
	if(discard) processShipForRegionStatsBack(&data[0], L, 1, t);

	return 0;
}

int main(int argc, char* argv[]){
	double start = getTime();
	
	for(int r = 0; r < REG; r++){
		regStat[r] = vector <float>(6LL * int(round(360 / regSize[r])) * (int(round(180 / regSize[r])) + 1), 0);
	}
	for(int f : hist) for(int h = 0; h < HIST; h++){
		histStat[f][h] = vector <float>(6 * (histSize[h] + 1), 0);
	}
	for(int f : hist2) for(int h = 0; h < HIST; h++){
		histStat2[f][h] = vector <float>(6 * (histSize[h] + 1), 0);
	}
	for(int f = 0; f < VAR2D; f++) for(int h = 0; h < HIST2D; h++){
		histStat2d[f][h] = vector <float>(6 * (histSize2d[h] + 1) * (histSize2d[h] + 1), 0);
	}
	
	for(int f : toBeSkipped) skipFeature[f] = 1;
	
	ifstream inl("lengths.csv");
	while(!inl.eof()){
		string line;
		getline(inl, line);
		if(!inl.eof()){
			vector <string> row = splitBy(line, ',');
			if(row.size() == 2){
				lengths[row[0]] = atoi(row[1].c_str());
			}
		}
	}
	inl.close();
	
	string phase = argc > 1 ? argv[1] : "train";
	training = phase == "train";
	string part = argc > 2 ? "_" + string(argv[2]) : "";
	int perc = argc > 2 ? atoi(argv[2]) : 100;
	int bin = 0; //argc > 3 ? atoi(argv[3]) : 0;
	if(argc > 3) nowRand = atoi(argv[3]);
	int seed = nowRand;
	
	if(phase != "train" && phase != "test" && phase != "rf"){
		cerr << "Incorrect phase name!" << endl;
		return 0;
	}
	string inputFile = perc == 100 && phase == "test" ? "testing.txt" : "training.txt";
	ifstream inp(inputFile.c_str());
	vector <string> files;
	vector <string> gt;
	while(!inp.eof()){
		string line;
		getline(inp, line);
		if(!inp.eof()){
			vector <string> row = splitBy(trim(line), ',');
			files.push_back(row[0]);
			if(row.size() > 1) gt.push_back(row[1]);
		}
	}
	inp.close();
		
	if(perc != 100){
		vector <string> files1, files2;
		vector <string> gt1, gt2;
		for(int i = 0; i < files.size(); i++){
			int binv = randBig() % 100;
			if(bin * (100 - perc) <= binv && binv < (bin + 1) * (100 - perc)){
				files2.push_back(files[i]);
				gt2.push_back(gt[i]);
			}
			else{
				files1.push_back(files[i]); 
				gt1.push_back(gt[i]); 
			}
		}
		if(phase == "train" || phase == "rf"){
			files = files1;
			gt = gt1;
		}
		else{
			files = files2;
			gt = gt2;
		}
	}
	
	if(phase == "rf"){
		ifstream in(("bindata" + SSTR(seed) + "/testdata" + part + ".dat").c_str(), ios::binary);
	    in.read((char*)&FEAT[0], FEATURES * SAMPLES * sizeof(float));
	    in.close();
	    for(int c = 1; c < CLASSES; c++){
		    in.open(("bindata" + SSTR(seed) + "/gt" + SSTR(c) + part + ".dat").c_str(), ios::binary);
		    in.read((char*)&RESULT[c][0], SAMPLES * sizeof(float));
		    in.close();
		}
	    featureScore.clear();
		featureScoreC.clear();
		featureScore.resize(FEATURES, 0);
		featureScoreC.resize(FEATURES, 0);
		for(int c = 1; c < CLASSES; c++){
			vector <Tree> randomForest(TREES);
			double tic = getTime();
			for(int j = 0; j < TREES; j++){
				randomForest[j] = buildTree(files.size(), c);
				if(j % 10 == 0) clog << j + 1 << " trees done...\r";
			}
			RFtoFile(randomForest, "bindata" + SSTR(seed) + "/rf" + SSTR(c) + part + ".dat");
		}
		ofstream pred("bindata" + SSTR(seed) + "/prediktors.txt");
		vector <pair <int,int> > stat;
		for(int i = 0; i < FEATURES; i++) stat.push_back(make_pair(featureScore[i], i));
		sort(stat.begin(), stat.end());
		int len = log10(stat.back().first + 0.1) + 2;
		vector <pair <int,int> > statC;
		for(int i = 0; i < FEATURES; i++) statC.push_back(make_pair(featureScoreC[i], i));
		sort(statC.begin(),statC.end());
		int lenC = log10(statC.back().first + 0.1) + 2;
		for(int i = FEATURES - 1; i >= 0; i--){
			pred << int2len(stat[i].first, len) << " " << string2len(SSTR(stat[i].second), 3) << "   |   " << int2len(statC[i].first, lenC) << " " << statC[i].second << endl;
		}
		pred.close();
		return 0;
	}
	
	if(phase == "train"){
		for(int i = 0; i < files.size(); i++){
			if(i % 10 == 0) clog << i << "\r";
			processShipForRegionStats(files[i], type[gt[i]]);
		}
		clog << endl;
		for(int i = 0; i < files.size(); i++){
			if(i % 10 == 0) clog << i << "\r";
			processShip(files[i], &FEAT[i], DISCARD, type[gt[i]]);
			for(int c = 1; c < CLASSES; c++){
				RESULT[c][i] = (type[gt[i]] == c) ? 1 : 0;
			}
		}
		ofstream out(("bindata" + SSTR(seed) + "/testdata" + part + ".dat").c_str(), ios::binary);
		out.write((const char *)&FEAT[0], FEATURES * SAMPLES * sizeof(float));
		out.close();
		for(int c = 1; c < CLASSES; c++){
			out.open(("bindata" + SSTR(seed) + "/gt" + SSTR(c) + part + ".dat").c_str(), ios::binary);
			out.write((const char *)&RESULT[c][0], SAMPLES * sizeof(float));
			out.close();
		}
		for(int r = 0; r < REG; r++){
			if(r != 0){
				out.open(("bindata" + SSTR(seed) + "/reg" + SSTR(r) + part + ".dat").c_str(), ios::binary);
				out.write((const char *)&regStat[r][0], ((long long) regStat[r].size()) * sizeof(float));
				out.close();
			}
			else{
				long long values = (long long) regStat[r].size();
				out.open(("bindata" + SSTR(seed) + "/reg" + SSTR(r) + part + ".dat").c_str(), ios::binary);
				out.write((const char *)&regStat[r][0], values / 2 * sizeof(float));
				out.close();			
				out.open(("bindata" + SSTR(seed) + "/regg" + SSTR(r) + part + ".dat").c_str(), ios::binary);
				out.write((const char *)&regStat[r][values / 2], values / 2 * sizeof(float));
				out.close();			
			}
		}
		for(int f : hist) for(int h = 0; h < HIST; h++){
			out.open(("bindata" + SSTR(seed) + "/hist" + SSTR(f) + "_" + SSTR(h) + part + ".dat").c_str(), ios::binary);
			out.write((const char *)&histStat[f][h][0], histStat[f][h].size() * sizeof(float));
			out.close();
		}
		for(int f : hist2) for(int h = 0; h < HIST; h++){
			out.open(("bindata" + SSTR(seed) + "/histt" + SSTR(f) + "_" + SSTR(h) + part + ".dat").c_str(), ios::binary);
			out.write((const char *)&histStat2[f][h][0], histStat2[f][h].size() * sizeof(float));
			out.close();
		}
		for(int f = 0; f < VAR2D; f++) for(int h = 0; h < HIST2D; h++){
			out.open(("bindata" + SSTR(seed) + "/hist2d" + SSTR(f) + "_" + SSTR(h) + part + ".dat").c_str(), ios::binary);
			out.write((const char *)&histStat2d[f][h][0], histStat2d[f][h].size() * sizeof(float));
			out.close();
		}
	}
	else{
		vector <Tree> randomForest[CLASSES];
		for(int c = 1; c < CLASSES; c++) randomForest[c] = RFfromFile("bindata" + SSTR(seed) + "/rf" + SSTR(c) + part + ".dat");
		for(int r = 0; r < REG; r++){
			if(r != 0){
			    ifstream in(("bindata" + SSTR(seed) + "/reg" + SSTR(r) + part + ".dat").c_str(), ios::binary);
			    in.read((char*)&regStat[r][0], ((long long) regStat[r].size()) * sizeof(float));
			    in.close();
			}
			else{
				long long values = (long long) regStat[r].size();
			    ifstream in(("bindata" + SSTR(seed) + "/reg" + SSTR(r) + part + ".dat").c_str(), ios::binary);
			    in.read((char*)&regStat[r][0], values / 2 * sizeof(float));
			    in.close();				
			    in.open(("bindata" + SSTR(seed) + "/regg" + SSTR(r) + part + ".dat").c_str(), ios::binary);
			    in.read((char*)&regStat[r][values / 2], values / 2 * sizeof(float));
			    in.close();				
			}
		}
		for(int f : hist) for(int h = 0; h < HIST; h++){
			ifstream in(("bindata" + SSTR(seed) + "/hist" + SSTR(f) + "_" + SSTR(h) + part + ".dat").c_str(), ios::binary);
			in.read((char *)&histStat[f][h][0], histStat[f][h].size() * sizeof(float));
			in.close();
		}
		for(int f : hist2) for(int h = 0; h < HIST; h++){
			ifstream in(("bindata" + SSTR(seed) + "/histt" + SSTR(f) + "_" + SSTR(h) + part + ".dat").c_str(), ios::binary);
			in.read((char *)&histStat2[f][h][0], histStat2[f][h].size() * sizeof(float));
			in.close();
		}
		for(int f = 0; f < VAR2D; f++) for(int h = 0; h < HIST2D; h++){
			ifstream in(("bindata" + SSTR(seed) + "/hist2d" + SSTR(f) + "_" + SSTR(h) + part + ".dat").c_str(), ios::binary);
			in.read((char *)&histStat2d[f][h][0], histStat2d[f][h].size() * sizeof(float));
			in.close();
		}
		double tic = getTime();
		vector <pair <double, int> > rank[5];
		for(int i = 0; i < files.size(); i++){
			if(i % 10 == 0) clog << i << "\r";
			vector <float> feat(FEATURES);
			processShip(files[i], &feat[0], false, -1);
			float* source = &feat[0];
			for(int c = 1; c < CLASSES; c++){
				double result = forestAssignResult(randomForest[c], source);
				cout << files[i] << "," << types[c] << "," << result << endl;
				if(perc != 100){
					rank[c].push_back(make_pair(result, i));
				}
			}
		}
		clog << endl;
		clog << "Average per ship: " << (getTime() - tic) / files.size() << " sec." << endl;
		if(perc != 100){
			vector <double> w = {0.0, 0.4, 0.3, 0.2, 0.1};
			vector <double> AuC(5, 0);
			for(int j = 1; j < 5; j++){
				int N_TPR = 0;
				int N_FPR = 0;
				for(int i = 0; i < files.size(); i++){
					if(gt[i] == types[j]) N_TPR++; else N_FPR++;
				}
				sort(rank[j].begin(), rank[j].end(), [&](pair <double, int> aa,pair <double, int> bb){return aa.first > bb.first;});
				double acc_si = 0;
				double acc_ni = 0;
				double FPR_prev = 0;
				for(int i = 0; i < files.size(); i++){
					int add = gt[rank[j][i].second] == types[j] ? 1 : 0;
					acc_si += add;
					acc_ni += 1 - add;
					double TPR = acc_si / N_TPR;
					double FPR = acc_ni / N_FPR;
					AuC[j] += TPR * (FPR - FPR_prev);
					FPR_prev = FPR;
				}
				clog << "AuC[" << j <<  "] = " << AuC[j] << endl;	
			}
			double WeightedAverage = w[0] * AuC[0] + w[1] * AuC[1] + w[2] * AuC[2] + w[3] * AuC[3] + w[4] * AuC[4];
			double score = max(1000000 * (2 * WeightedAverage - 1), 0.0);
			clog << "Score = " <<  score << endl;
			ofstream of("score.csv", ios_base::app);
			of << SSTR(seed) + "," + SSTR(score) + "\n"; 
			of.close();
		}
	}
	
	clog << getTime() - start << " seconds" << endl;
	
	return 0;
}

