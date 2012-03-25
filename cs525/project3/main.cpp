/*******************************************************************************
*
* Program #3 -- Arrow Shooting Game
*
* DESC: This is a game implemented in C/C++ and OpenGL which allows a player to
*	shoot arrows at a moving target.  The target moves at random speeds which
*   change randomly three times per transit of the play area.
*
* ARCHITECTURE: This program is designed around the use of a single static game
*	state struct (theGame) which contains sub-structs which contain information
*	about specific game objects (target, arrow, shooter).  Output is handled by
*	special functions which draw the game objects based on the state data.
*	Game logic is spread through-out various functions, mostly concentrated in
*   idle_func() which is bound to the OGL Idle event.
*
*	The draw_func() is a modal function which draws the game screen.  The game
*	screen is broken up into two areas, a game area and a status area.  Both
*	are stored as static bounding rects, game_area and status_area respectivly.
*	draw_func() is modal on the game state, when the game is active it draws the
*	current frame.  When the game is ended it draws a final message.
*
*	A number of generic library functions are defined in lib.h.  These include
*	a method to do GLUT text output and to draw rectangles as a filled polygon.
*	Also there's a simple color library and pre-defined color palette defined
*	as consts.  All colors are RGBA and alpha blending has been enabled, but is
*	unused in this program.
*
* PROGRAM CONTROLS:
*	u/U -- Move player up.
*	j/J -- Move player down.
* 	n/N -- Start new game.
*	q/Q -- Quit program.
*
*******************************************************************************/
#include <iostream>
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include "cs_425_525_setup.h"
#include "lib.h"

#define WINDOW_X 500
#define WINDOW_Y 600
#define WINDOW_NAME "Program 3 -- Arrow Shooting Game"
#define PI 3.14159265
#define SPEED_LOWER_BOUND 50
#define SPEED_UPPER_BOUND 300
#define SHOOTER_MOVE 10
#define ARROW_SPEED 120
#define COLL_GREEN_TARGET 1
#define COLL_RED_TARGET 2
#define SHOTS_PER_ROUND 5
#define ROUND_TIME 7

#define RANDRANGE(a,b) ((a) + (rand() % ((b) - (a)) + 1))
#define SWAP(a,b) ((a) ^= (b) ^= (a) ^= (b))

typedef struct {
	int y;			// y-coord of center mass
	int speed;		// speed in pixels / second
	int direction;	// direction of travel (1 or -1)
	int distance;	// distance left in pixels, for current speed
	int changed;	// times speed has changed this transit
} target_t;

typedef struct {
	int y;			// y-coord of center mass
} shooter_t;

typedef struct {
	int 	x;			// x-coord of center mass
	int 	y;			// y-coord of center mass
	bool 	is_shot;	// set to true if arrow was shot by the shooter
	int 	shots_left;	// # of shots left
} arrow_t;

typedef struct {
	target_t	target;		// the target
	shooter_t	shooter;	// the shooter
	arrow_t		arrow;		// the arrow
	int			score;		// the current score
	clock_t 	last_clock;	// the last clock value (in ticks)
	float		round_time;	// the round time left in seconds
	bool		is_end;		// true if curr game ended
} game_t;

typedef struct {
	long int	total_frames;	// the total # of frames drawn
	float		fps;			// instantaneous frames / sec
	float		avg_fps;		// average frames / sec
	float		play_time;		// total seconds in game
	float		idle_time;		// total seconds idle (at end screen)
	int			arrows_shot;	// count of arrows shot
	int			red_hits;		// count of red hits
	int			green_hits;		// count of green hits
	int			auto_shot;		// times the arrow was auto-shot because of timer
	int			games_played;	// number of games played
	int			total_score;	// sum of all scores
} stats_t;

void init_stats(void);
void init_game(void);
void display_func(void);
void keyboard_func(unsigned char, int, int);
void mouse_func(int, int, int, int);
void idle_func(void);
void terminate(void);
void dump_stats(void);

void make_bounding_box(rect_t*, int, int, int, int);
int collision_detect(void);

void target_change_speed(void);
void target_change_direction(void);
void target_draw(void);
void shooter_draw(void);
void arrow_shoot(void);
void arrow_reset(void);
void arrow_draw(void);
void game_end(void);
void game_reset(void);

const rect_t status_area = { 600, 0, 500, 500 };
const rect_t game_area = { 499, 0, 0, 500 };
static game_t theGame;
static stats_t theStats;

int main(int argc, char* argv[]) {

	// start up the game
	init_stats();
	init_game();

	// setup OpenGL
	glutInit(&argc,argv);
	init_setup(WINDOW_X,WINDOW_Y,WINDOW_NAME);
	glutDisplayFunc(display_func);
	glutKeyboardFunc(keyboard_func);
	glutIdleFunc(idle_func);

	// run the OGL main loop
	glutMainLoop();

	return 0;
}

void init_stats(void) {
	// zero out theStats object
	memset(&theStats,0,sizeof(stats_t));
}

void init_game(void) {
	// seed the random number generator
	srand(time(NULL));

	// zero out theGame object
	memset(&theGame,0,sizeof(game_t));

	// get the initial clock timing
	theGame.last_clock = clock();

	// reset up the game
	game_reset();
}

void display_func(void) {
	char buffer[128];

	// black background
	glClearColor(BLACK.r, BLACK.g, BLACK.b, BLACK.a);
	glClear(GL_COLOR_BUFFER_BIT);

	// if the game is in the end state do special output
	if (theGame.is_end) {

		// write game over message
		memset(buffer,0,sizeof(char)*128);
		sprintf(buffer,"GAME OVER");
		print_string(status_area.top - 60, status_area.left + 20,
			buffer, GLUT_BITMAP_HELVETICA_18, WHITE);

		// write the final score
		memset(buffer,0,sizeof(char)*128);
		sprintf(buffer,"Final score: %8.4d", theGame.score);
		print_string(status_area.top - 80, status_area.left + 20,
			buffer, GLUT_BITMAP_HELVETICA_18, RED);

		// write what to do next
		memset(buffer,0,sizeof(char)*128);
		sprintf(buffer,"To exit press 'q'.  To play again press 'n'");
		print_string(status_area.top - 100, status_area.left + 20,
			buffer, GLUT_BITMAP_HELVETICA_18, WHITE);

	} else {
		// draw the white status box
		draw_rect(status_area,WHITE);

		// draw the target
		target_draw();

		// draw the shooter
		shooter_draw();

		// draw the arrow
		arrow_draw();

		// write the game title
		memset(buffer, 0, sizeof(char) * 128);
		sprintf(buffer, "Archery Game");
		print_string(status_area.top - 20, status_area.left + 30,
			buffer, GLUT_BITMAP_HELVETICA_18, DK_GRAY);

		// write the shots left counter
		memset(buffer, 0, sizeof(char) * 128);
		sprintf(buffer, "Round: %8.2d", theGame.arrow.shots_left);
		print_string(status_area.top - 20, status_area.right - 150,
			buffer, GLUT_BITMAP_HELVETICA_18, DK_GRAY);

		// write the shot timer
		memset(buffer, 0, sizeof(char) * 128);
		sprintf(buffer, "Timer: %8.2f", theGame.round_time);
		print_string(status_area.bottom + 60, status_area.left + 30,
			buffer, GLUT_BITMAP_HELVETICA_18, DK_GRAY);

		// write the score
		memset(buffer, 0, sizeof(char) * 128);
		sprintf(buffer, "Score: %8.4d", theGame.score);
		print_string(status_area.bottom + 60, status_area.right - 150,
			buffer, GLUT_BITMAP_HELVETICA_18, DK_GRAY);

		// write the instant & avg fps
		memset(buffer, 0, sizeof(char) * 128);
		sprintf(buffer, "Inst. FPS: %8.2f", theStats.fps);
		print_string(status_area.bottom + 40, status_area.left + 30,
			buffer, GLUT_BITMAP_HELVETICA_18, DK_GRAY);
		memset(buffer, 0, sizeof(char) * 128);
		sprintf(buffer, "Avg. FPS: %6.2f", theStats.avg_fps);
		print_string(status_area.bottom + 40, status_area.right - 150,
			buffer, GLUT_BITMAP_HELVETICA_18, DK_GRAY);
	}

	// count the frame for end stats
	theStats.total_frames++;

	// flush & swap the buffer
	glFlush();
	glutSwapBuffers();
}

void keyboard_func (unsigned char c, int x, int y) {
	// switch on the keyboard character pressed
	switch (c) {
		case 'U':
		case 'u': { // move shooter up
			theGame.shooter.y += SHOOTER_MOVE;
			theGame.arrow.y += SHOOTER_MOVE;

			// keep the shooter in bounds
			if (theGame.shooter.y > game_area.top - 12) {
				theGame.shooter.y = game_area.top - 12;
				theGame.arrow.y = game_area.top - 12;
			}
		} break;
		case 'J':
		case 'j': { // move shooter down
			theGame.shooter.y -= SHOOTER_MOVE;
			theGame.arrow.y -= SHOOTER_MOVE;

			// keep the shooter in bounds
			if (theGame.shooter.y < game_area.bottom + 12) {
				theGame.shooter.y = game_area.bottom + 12;
				theGame.arrow.y = game_area.bottom + 12;
			}
		} break;
		case ' ': { // shoot the arrow
			arrow_shoot();
		} break;
		case 'N':
		case 'n': {
			game_reset();
		} break;
		case 'Q': // quit the program
		case 'q': {
			terminate();
			exit(0);
		} break;
	}
}

void idle_func(void) {
	clock_t curr_clock = clock();
	double elapsed_sec = (double)(curr_clock - theGame.last_clock) / CLOCKS_PER_SEC;
	int target_move = (int)ceil(theGame.target.speed * elapsed_sec);
	int arrow_move = (int)ceil(ARROW_SPEED * elapsed_sec);

	// short-circut the idle logic if the game is ended
	if (theGame.is_end) {
		// update the last clock to the curr clock
		theGame.last_clock = curr_clock;

		// do some stats recording
		theStats.idle_time += elapsed_sec;

		// force a redraw
		glutPostRedisplay();

		return;
	}

	// update the target y coord
	theGame.target.y = theGame.target.y +
		(theGame.target.direction * target_move);
	theGame.target.distance -= target_move;

	// keep the target in bounds
	if (theGame.target.y < 25) {
		theGame.target.y = 25;
		target_change_direction();
	}
	if (theGame.target.y > game_area.top - 25) {
		theGame.target.y = game_area.top - 25;
		target_change_direction();
	}

	// check to see if it's time to change the target speed
	if (theGame.target.distance <= 0)
		target_change_speed();

	// update the round timer if the arrow hasn't been shot
	if (!theGame.arrow.is_shot) {
		theGame.round_time -= elapsed_sec;

		// shoot arrow if timer expired
		if (theGame.round_time <= 0.0) {
			theGame.round_time = 0.0;
			arrow_shoot();
			theStats.auto_shot++;
		}
	}

	// do the arrow stuff if it's in motion
	if (theGame.arrow.is_shot) {
		theGame.arrow.x -= arrow_move;

		if (theGame.arrow.x < game_area.left - 20) {
			if (theGame.arrow.shots_left <= 0)
				game_end();
			else
				arrow_reset();
		} else {
			switch (collision_detect()) {
				case COLL_RED_TARGET: {
					theGame.score += 5;
				}
				case COLL_GREEN_TARGET: {
					theGame.score += 5;
					if (theGame.arrow.shots_left <= 0)
						game_end();
					else
						arrow_reset();
				} break;
				default: break;
			}
		}
	}

	// update the last clock to the curr clock
	theGame.last_clock = curr_clock;

	// calc the instaneous fps
	theStats.fps = (float)(1.0 / elapsed_sec);
	theStats.avg_fps += theStats.fps;
	theStats.avg_fps /= 2;

	// do some stats recording
	theStats.play_time += elapsed_sec;

	// force a redraw
	glutPostRedisplay();

}

void terminate(void) {
	dump_stats();
}

void dump_stats(void) {
	cout << "Graphics Stats:" << endl
		 << "\tTotal Frames = " << theStats.total_frames << endl
		 << "\tOverall FPS  = " <<
			((float)theStats.total_frames) /
			(theStats.play_time + theStats.idle_time) << endl;
	cout << "Play Stats:" << endl
		 << "\tGames Played = " << theStats.games_played << endl
		 << "\tTotal Points scored = " << theStats.total_score << endl
		 << "\tTime Spent Playing = " << theStats.play_time << endl
		 << "\tArrows Shot = " << theStats.arrows_shot << endl
		 << "\tAccuracy = " <<
			(((float)(theStats.red_hits + theStats.green_hits)) /
			theStats.arrows_shot) * 100 << "%" << endl
		 << "\t\tRed Accuracy = " <<
			(((float)theStats.red_hits) /
			theStats.arrows_shot) * 100 << "%" << endl
		 << "\t\tGreen Accuracy = " <<
			(((float)theStats.green_hits) /
			theStats.arrows_shot) * 100 << "%" << endl
		 << "\tAuto-shot Arrows = " << theStats.auto_shot << endl;
}

void make_bounding_box(rect_t* bb, int x, int y, int width, int height) {
	int offset_x = (int)floor(width / 2);
	int offset_y = (int)floor(height / 2);

	// clear the bounding box struct
	memset(bb,0,sizeof(rect_t));

	// calc the bounding box top-left and bottom-right
	bb->top = y + offset_y - ((height % 2 == 0) ? 1 : 0);
	bb->left = x - offset_x;
	bb->bottom = y - offset_y;
	bb->right = x + offset_x - ((width % 2 == 0) ? 1 : 0);
}

void target_change_speed(void) {
	int dist_to_boundry = 0;
	int lower_bound = 0;
	int upper_bound = 0;

	// calc a new random speed in the bounds
	theGame.target.speed = RANDRANGE(SPEED_LOWER_BOUND, SPEED_UPPER_BOUND);

	// calc the distance to the boundry it's going to hit
	if (theGame.target.direction > 0)
		dist_to_boundry = game_area.top - theGame.target.y;
	else
		dist_to_boundry = theGame.target.y;

	// set the upper/lower bounds for the random generation for the new dist.
	lower_bound = (int)ceil(theGame.target.y / 4);
	upper_bound = (int)floor(dist_to_boundry / 2);

	// swap the bounds if the lower_bound is greater then the upper bound
	if (lower_bound > upper_bound)
		SWAP(lower_bound, upper_bound);

	// generate the distance to travel at this new speed, if two changes have
	// occured its the distance to the boundry.
	if (lower_bound != upper_bound)
		theGame.target.distance = RANDRANGE(lower_bound,upper_bound);
	else
		theGame.target.distance = dist_to_boundry;

	// if we have the special case where it constantly used values close to the
	// lower bound we need to make sure we only have two changed speed per transit
	if (theGame.target.changed > 2)
		theGame.target.distance = dist_to_boundry;

	// update the changed speed counter
	theGame.target.changed++;
}

void target_change_direction(void) {
	theGame.target.direction *= -1;
	target_change_speed();
	theGame.target.changed = 0;
}

void target_draw(void) {
	rect_t target_green, target_red;

	// create the draw box (which also happens to be the bounding box)
	make_bounding_box(&target_green,
		game_area.left + 15, theGame.target.y, 10, 50);
	make_bounding_box(&target_red,
		game_area.left + 26, theGame.target.y, 10, 30);

	// output the created rectangle
	draw_rect(target_green,GREEN);
	draw_rect(target_red,RED);
}

void shooter_draw(void) {
	rect_t shooter;

	// create the draw box (which also happens to be the bounding box)
	make_bounding_box(&shooter,game_area.right - 22, theGame.shooter.y, 25, 25);

	// output the created rectangle
	draw_rect(shooter, LT_GREEN);
}

void arrow_shoot(void) {
	theStats.arrows_shot++;
	theGame.arrow.is_shot = true;
	theGame.arrow.shots_left--;
}

void arrow_reset(void) {
	theGame.arrow.x = game_area.right - 45;
	theGame.arrow.y = theGame.shooter.y;
	theGame.arrow.is_shot = false;
	theGame.round_time = ROUND_TIME;
}

void arrow_draw(void) {
	rect_t arrow;

	// create the draw box (which also happens to be the bounding box)
	make_bounding_box(&arrow,theGame.arrow.x, theGame.arrow.y, 40, 3);

	// output the created rectangle
	draw_rect(arrow, BROWN);
}

int collision_detect(void) {
	rect_t target_green, target_red;
	int y = theGame.arrow.y;
	int x = theGame.arrow.x - 20;

	// create the bounding boxes for the green and red targets
	make_bounding_box(&target_green,
		game_area.left + 15, theGame.target.y, 10, 50);
	make_bounding_box(&target_red,
		game_area.left + 26, theGame.target.y, 10, 30);

	// figure out if the arrow hit the red box
	if (target_red.top >= y && target_red.bottom <= y)
		if (target_red.left <= x && target_red.right >= x) {
			theStats.red_hits++;
			return COLL_RED_TARGET;
		}

	// figure out if the arrow hit the green box
	if (target_green.top >= y && target_green.bottom <= y)
		if (target_green.left <= x && target_green.right >= x) {
			theStats.green_hits++;
			return COLL_GREEN_TARGET;
		}

	return 0;
}

void game_end(void) {
	theStats.games_played++;
	theStats.total_score += theGame.score;
	theGame.is_end = true;
}

void game_reset(void) {
	// reset the is_end flag
	theGame.is_end = false;

	// setup the target
	theGame.target.y = (int)floor(abs((game_area.top - game_area.bottom) / 2));
	theGame.target.direction = 1;  // ascend
	target_change_speed();
	theGame.target.changed = 0;

	// setup the shooter
	theGame.shooter.y = (int)floor(abs((game_area.top - game_area.bottom) / 2));

	// setup the inital arrow
	theGame.arrow.x = game_area.right - 45;
	theGame.arrow.y = theGame.shooter.y;
	theGame.arrow.is_shot = false;
	theGame.arrow.shots_left = SHOTS_PER_ROUND;

	// set the inital round timer
	theGame.round_time = ROUND_TIME;
}
