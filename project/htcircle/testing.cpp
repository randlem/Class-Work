#define DEBUG		1
#define USAGE_MESSAGE "Usage: htctesting [file prefix]"
#define BIN_THRLD   0.75
#define STAT_RUNS   100

#define IMG(x,y)     img.pixels[(y)][(x)]
#define IMG_MAP(x,y) img_map[(y) * in_img.width + (x)]
#define ACCUM(a,b,r) accum[(a)][(b)][(r)]
#define RANGE(rng)   (double)(rng.high - rng.low)

#include <iostream>
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <unistd.h>
#include <time.h>
#include <math.h>
#include "util.h"
#include "gfxutil.h"
#include "imgutil.h"

typedef struct {
	double a;
	double b;
	double r;
	uint_t bin_cnt;
} circle_t;

typedef struct {
	double low;
	double high;
} range_t;

typedef struct {
	double bytes;
	uint_t count;
	uint_t filled;
	uint_t bin_max;
	double bin_mean;
} stats_accum_t;

typedef struct {
	double ht_circle;
	double identify_circle;
} stats_timing_t;

typedef struct {
	int             runs;
	stats_accum_t   accum;
	stats_timing_t* timing;
} stats_t;

const range_t r_range = {5.0, 125.0};
const range_t a_range = {-125.0, 375.0};
const range_t b_range = {-125.0, 375.0};

string    file_prefix = "";
string    in_file     = "";
string    out_file    = "";
string    stats_file  = "";
img_t     in_img;
img_t     out_img;
uchar_t*  img_map     = NULL;
uint_t*** accum	      = NULL;
stats_t   stats;
circle_t* circles     = NULL;
uint_t    num_circles;

uint_t a_bins = 0;
uint_t b_bins = 0;
uint_t r_bins = 0;
uint_t accum_size = 0;

void setup(int*, char**);
void terminate();
void collect_accum_stats();
void stats_to_csv();
double time_diff(timespec*, timespec*);

void plot_point(img_t*, uint_t, uint_t, const color_t &);
void bresenham_circle(img_t*, uint_t, uint_t, uint_t, const color_t &);
void draw_ided_circles(img_t*, const color_t &);

void translate_img_to_binary(img_t*);
void allocate_accum();
void deallocate_accum();
void clear_accum();
void ht_circle();
int identify_circle();

int main (int argc, char *argv[]) {
	int i;
	timespec start, end;

	// setup the program
	setup(&argc,argv);
	if (!image_read(in_file,&in_img))
		return 1;
	translate_img_to_binary(&in_img);
	allocate_accum();

	// do timed runs.  we only care about timing ht_circle() and identify_circle()
	cout << "Doing " << stats.runs << " timed runs...";
	cout.flush();

	for (i=0; i < stats.runs; i++) {
		memset(&stats.timing[i],0,sizeof(stats_timing_t));
		if ((i % 10) == 0) {
			cout << i << "..";
			cout.flush();
		}

		clock_gettime(CLOCK_REALTIME,&start);
		ht_circle();
		clock_gettime(CLOCK_REALTIME,&end);
		stats.timing[i].ht_circle = time_diff(&start,&end);

		clock_gettime(CLOCK_REALTIME,&start);
		num_circles = identify_circle();
		clock_gettime(CLOCK_REALTIME,&end);
		stats.timing[i].identify_circle = time_diff(&start,&end);

		clear_accum();

		delete [] circles;
		circles = NULL;
		num_circles = 0;
	}
	cout << endl;

	// collect accum stats and draw some identified circles
	image_copy(&in_img,&out_img);
	draw_ided_circles(&out_img,ORANGE);
	image_write(out_file,&out_img);
	collect_accum_stats();

	// dump the gathered accum stats
	cout << "Accum stats:" << endl;
	cout << "\tsize (in KB) = " << (int)(stats.accum.bytes / 1024) << endl;
	cout << "\tcells = " << stats.accum.count << endl;
	cout << "\tfilled = " << stats.accum.filled << endl;
	cout << "\tmax = " << stats.accum.bin_max << endl;
	cout << "\tmean = " << stats.accum.bin_mean << endl;

	deallocate_accum();

	return 0;
}

void setup (int* argc, char **argv) {
	img_map = NULL;
	accum   = NULL;

	// grab the cmd line arguments
	if (*argc != 2) {
		cerr << USAGE_MESSAGE << endl;
		exit(-1);
	}

	// create the filenames
	file_prefix = argv[1];
	in_file     = file_prefix + "_in.png";
	out_file    = file_prefix + "_out.png";
	stats_file  = file_prefix + "_stats.csv";
	debug("file prefix %s, in file = %s, out file = %s, stats = %s",file_prefix.c_str(),in_file.c_str(),out_file.c_str(),stats_file.c_str());

	// clear out and init the stats struct
	memset(&stats,0,sizeof(stats_t));
	stats.runs   = STAT_RUNS;
	stats.timing = new stats_timing_t[stats.runs];
}

void terminate () {
	// cleanup stats memory
	if (stats.timing != NULL) {
		delete [] stats.timing;
		stats.timing = NULL;
	}

	// cleanup the image map memory
	if (img_map != NULL) {
		delete [] img_map;
		img_map = NULL;
	}

	// cleanup the accum memory
	if (accum != NULL) {
		delete [] accum;
		accum = NULL;
	}
}

double time_diff(timespec* start, timespec* end) {
	return 0;
}

void collect_accum_stats () {
	int a_bin,b_bin,r_bin,curr;

	stats.accum.count = a_bins * b_bins * r_bins;
	stats.accum.bytes = stats.accum.count * sizeof(uint_t);

	for (a_bin=0; a_bin < a_bins; a_bin++) {
		for (b_bin=0; b_bin < b_bins; b_bin++) {
			for (r_bin=0; r_bin < r_bins; r_bin++) {
				curr = ACCUM(a_bin,b_bin,r_bin);
				if (curr > 0) {
					stats.accum.filled++;
					if (stats.accum.bin_max < curr)
						stats.accum.bin_max = curr;
					stats.accum.bin_mean += curr;
				}
			}
		}
	}
	stats.accum.bin_mean = stats.accum.bin_mean / stats.accum.filled;
}

void plot_point (img_t* img, uint_t x, uint_t y, const color_t &c) {
	img->pixels[y][x].rgba = c.rgba;
}

void bresenham_circle (img_t* img, uint_t a, uint_t b, uint_t r, const color_t &c) {
	int y    = r;
	int d    = -r;
	int x2m1 = -1;
	int x;

	plot_point(img, a, b + r, c);
	plot_point(img, a, b - r, c);
	plot_point(img, a + r, b, c);
	plot_point(img, a - r, b, c);
	for(x=1; x < r / sqrt(2); x++) {
		x2m1 += 2;
		d += x2m1;

		if (d >= 0) {
			y--;
			d -= (y<<1);
		}

		plot_point(img, a + x, b + y, c);
		plot_point(img, a + x, b - y, c);
		plot_point(img, a - x, b + y, c);
		plot_point(img, a - x, b - y, c);
		plot_point(img, a + y, b + x, c);
		plot_point(img, a + y, b - x, c);
		plot_point(img, a - y, b + x, c);
		plot_point(img, a - y, b - x, c);
	}
}

void draw_ided_circles(img_t* img, const color_t &c) {
	for (int i=0; i < num_circles; i++) {
		bresenham_circle(img,(uint_t)circles[i].a,
			(uint_t)circles[i].b,(uint_t)circles[i].r,c);
	}
}

void translate_img_to_binary (img_t* img) {
	int y,x,map_size;

	map_size = img->height * img->width;
	img_map = new uchar_t[map_size];
	memset(img_map,0,map_size * sizeof(uchar_t));

	for (y=0; y < img->height; y++) {
		for (x=0; x < img->width; x++) {
			img_map[y * img->width + x] = img->pixels[y][x].rgba;
		}
	}
}

void allocate_accum () {
	int i,j;

	a_bins     = (uint_t)RANGE(a_range);
	b_bins     = (uint_t)RANGE(b_range);
	r_bins     = (uint_t)RANGE(r_range);
	accum_size = a_bins * b_bins * r_bins;

	accum      = new uint_t**[a_bins];
	for(i=0; i < a_bins; i++) {
		accum[i] = new uint_t*[b_bins];
		for(j=0; j < b_bins; j++) {
			accum[i][j] = new uint_t[r_bins];
			memset(accum[i][j],0,r_bins*sizeof(uint_t));
		}
	}

	//debug("Accum: size = %d",accum_size);
	//debug("A: range = (%0.2f,%0.2f) bins = %d",a_range.low,a_range.high,a_bins);
	//debug("B: range = (%0.2f,%0.2f) bins = %d",b_range.low,b_range.high,b_bins);
	//debug("R: range = (%0.2f,%0.2f) bins = %d",r_range.low,r_range.high,r_bins);
}

void deallocate_accum () {
	int i,j;

	for(i=0; i < a_bins; i++) {
		for(j=0; j < b_bins; j++) {
			delete [] accum[i][j];
		}
		delete [] accum[i];
	}
	delete [] accum;

	a_bins = 0;
	b_bins = 0;
	r_bins = 0;
	accum_size = 0;
}

void clear_accum() {
	int i,j;

	for(i=0; i < a_bins; i++) {
		for(j=0; j < b_bins; j++) {
			memset(accum[i][j],0,r_bins*sizeof(uint_t));
		}
	}
}

void ht_circle () {
	int y,x,a_bin,b_bin,r_bin;
	double a,b,r,theta,a_pitch,b_pitch,r_pitch;

	a_pitch = RANGE(a_range) / a_bins;
	b_pitch = RANGE(b_range) / b_bins;
	r_pitch = RANGE(r_range) / r_bins;

	for (y=0; y < in_img.height; y++) {
		for (x=0; x < in_img.width; x++) {
			if (IMG_MAP(x,y) > 0) {
				for (r_bin=(int)r_range.low; r_bin < r_bins; r_bin++) {
					for (theta=0.0; theta < 360.0; theta++) {
						r = r_bin * r_pitch;
						a = x + (r * cos((theta * 2.0 * M_PI) / 360.0));
						b = y + (r * sin((theta * 2.0 * M_PI) / 360.0));
						a_bin = (int)(a / a_pitch) + (int)fabs(a_range.low);
						b_bin = (int)(b / b_pitch) + (int)fabs(b_range.low);
						ACCUM(a_bin, b_bin, r_bin)++;
					}
				}
			}
		}
	}
}

int identify_circle () {
	int threshold,a_bin,b_bin,r_bin,bin_max,filled,circles_cnt,max_circles;
	double a_pitch, b_pitch, r_pitch;

	bin_max = filled = 0;
	for (a_bin=0; a_bin < a_bins; a_bin++) {
		for (b_bin=0; b_bin < b_bins; b_bin++) {
			for (r_bin=0; r_bin < r_bins; r_bin++) {
				filled++;
				if (bin_max < ACCUM(a_bin,b_bin,r_bin))
					bin_max = ACCUM(a_bin,b_bin,r_bin);
			}
		}
	}

	a_pitch     = RANGE(a_range) / a_bins;
	b_pitch     = RANGE(b_range) / b_bins;
	r_pitch     = RANGE(r_range) / r_bins;
	threshold   = (int)ceil(bin_max * BIN_THRLD);
	circles_cnt = 0;
	max_circles = (int)(filled * 0.01);
	circles     = new circle_t[max_circles];
	if (circles == NULL) {
		debug("Failed to allocate circles array.");
		return 0;
	}
	memset(circles,0,max_circles * sizeof(circle_t));

	for (a_bin=0; a_bin < a_bins; a_bin++) {
		for (b_bin=0; b_bin < b_bins; b_bin++) {
			for (r_bin=0; r_bin < r_bins; r_bin++) {
				if (ACCUM(a_bin,b_bin,r_bin) >= threshold) {
					circles[circles_cnt].a       = (a_bin - fabs(a_range.low)) * a_pitch;
					circles[circles_cnt].b       = (b_bin - fabs(b_range.low)) * b_pitch;
					circles[circles_cnt].r       = r_bin * r_pitch;
					circles[circles_cnt].bin_cnt = ACCUM(a_bin,b_bin,r_bin);
					circles_cnt++;
					max_circles--;
				}
				if (max_circles < 0) {
//					debug("Hit max circles threshold.");
					goto end;
				}
			}
		}
	}

	end:
	return circles_cnt;
}
