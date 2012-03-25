#include "lib.h"

#ifndef __BPM_H__
#define __BPM_H__

#define BPM_TIMER_PULSES 5

typedef struct {
	float	bpm;
	bool	timer_active;
	int		timer_count;
	clock_t	timer_last;
	clock_t	timer_elp_avg;
	bool	flasher_state;
	clock_t	flasher_threshold;
	float	flasher_on_sec;
	float	flasher_off_sec;
	rect_t	flasher_rect;
} bpm_t;

static bpm_t	global_bpm;

void bpm_init(void) {
	memset(&global_bpm,0,sizeof(bpm_t));
	global_bpm.timer_active			= false;
	global_bpm.flasher_state		= false;
	global_bpm.flasher_rect.top		= 20;
	global_bpm.flasher_rect.bottom	= 5;
	global_bpm.flasher_rect.left	= 15;
	global_bpm.flasher_rect.right	= 30;
}

void bpm_reset_pulse_counter(void) {
	global_bpm.timer_count	= 0;
	global_bpm.timer_last	= clock();
	global_bpm.timer_active	= true;
}

void bpm_count_pulse(void) {
	clock_t	curr_clock;
	float	elapsed_secs;

	if (global_bpm.timer_count > BPM_TIMER_PULSES)
		return;

	if (global_bpm.timer_count == 0) {
		global_bpm.timer_last = clock();
		global_bpm.timer_count++;
	} else {
		curr_clock = clock();
		global_bpm.timer_elp_avg += curr_clock - global_bpm.timer_last;
		global_bpm.timer_elp_avg /= 2;
		global_bpm.timer_last = curr_clock;
	}

	if (global_bpm.timer_count == BPM_TIMER_PULSES) {
		global_bpm.timer_active = false;
		global_bpm.bpm = (60.0 * CLOCKS_PER_SEC) /
									global_bpm.timer_elp_avg;
		global_bpm.flasher_state = false;
		global_bpm.flasher_threshold = clock() +
			(global_bpm.flasher_off_sec * CLOCKS_PER_SEC);
		global_bpm.flasher_off_sec = (60.0 / global_bpm.bpm) * 0.2;
		global_bpm.flasher_on_sec = (60.0 / global_bpm.bpm) * 0.8;
	}

	global_bpm.timer_count++;
}

void bpm_animate_flasher(void) {
	if (global_bpm.flasher_threshold < clock()) {
		if (global_bpm.flasher_state) {
			global_bpm.flasher_state = false;
			global_bpm.flasher_threshold = clock() +
				(global_bpm.flasher_off_sec * CLOCKS_PER_SEC);
		} else {
			global_bpm.flasher_state = true;
			global_bpm.flasher_threshold = clock() +
				(global_bpm.flasher_on_sec * CLOCKS_PER_SEC);
		}
	}
}

void bpm_draw_flasher(void) {
	if (!global_bpm.timer_active && floor(global_bpm.bpm) != 0) {
		if (global_bpm.flasher_state) {
			draw_rect_filled(global_bpm.flasher_rect,GREEN);
		} else {
			draw_rect_hollow(global_bpm.flasher_rect,DK_GRAY);
		}
	}
}

void bpm_draw(void) {
	char buffer[15];

	memset(buffer,0,sizeof(char)*15);
	sprintf(buffer,"BPM: %.2f",global_bpm.bpm);
	print_string(5, 35, buffer, GLUT_BITMAP_HELVETICA_18, WHITE);

	bpm_draw_flasher();
}

#endif
