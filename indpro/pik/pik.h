#ifndef __PIK_H__
#define __PIK_H__

enum state { WALK, EAT, TALK };
typedef state;

typedef struct {
	int id;
	state curr_state;
	int x;
	int y;
	int time_since_eat;
} pik;

void init_pik(pik* da_pik, int id, int x, int y);
void update_pik(pik* da_pik, pik* nearest_pik, int dist);
void print_stats();
void pack_pik(pik* da_pik, char* buffer, int size_buffer);
void unpack_pik(pik* da_pik, char* buffer, int size_buffer);
void gather_stats(pik* piks, int count);

#endif
