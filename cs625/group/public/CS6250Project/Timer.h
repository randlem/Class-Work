/**
 * Timer class definition file.
 *
 * Provides basic timer functionality which calculates running time for certain
 * part of your program. Very useful for analyzing algorithm running time.
 * After a Timer object is created, calling the Start() member function starts 
 * the timer running. The Stop() member function stops the timer. Once the 
 * timer is stopped, Start() must be called again to reset the timer. 
 * GetElapsedSeconds() returns the number of seconds elapsed. Calling 
 * GetElapsedSeconds() while the timer is still running gives you the elapsed
 * time up to that point.
 *
 * Note: 
 * This Timer class implementation is platform independent. The accuracy of the 
 * timer might differ based on the platform/hardware. Both *NIX/Windows version 
 * provides high resolution up to one microseconds. For other platforms the timer 
 * from standard library is used instead. It is not as precise, but is 
 * supported across all platforms.
 */
#ifndef TIMER_H
#define TIMER_H

#if defined(linux) || defined(__linux) || defined(__linux__) || defined(__unix__) || defined(unix) || defined(__unix)
    #define __PLATFORM_UNIX__
    #include <sys/time.h>
    #define stopwatch timeval
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    #define __PLATFORM_WINDOWS__
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #define stopwatch LARGE_INTEGER
#else
    #define __PLATFORM_OTHER__
    #include <ctime>
    #define stopwatch clock_t
#endif


class Timer
{
public:
    Timer();
    void Start();
    void Stop();
    double GetElapsedSeconds();

private:
    stopwatch begin;
    stopwatch end;
    bool isRunning;
    
#ifdef __PLATFORM_WINDOWS__
    DWORD mStartTick;
	LONGLONG mLastTime;
    LARGE_INTEGER mFrequency;
    DWORD mTimerMask;
#endif
    
    void SetBeginTime();
    void SetEndTime();
};

Timer::Timer()
    : isRunning(false)
{
}

void Timer::Start()
{
    isRunning = true;
    SetBeginTime();
}

void Timer::Stop()
{
    isRunning = false;
    SetEndTime();
}


#ifdef __PLATFORM_UNIX__


void Timer::SetBeginTime()
{
    gettimeofday(&begin, NULL);
}

void Timer::SetEndTime()
{
    gettimeofday(&end, NULL);
}

double Timer::GetElapsedSeconds()
{
    if (isRunning)
        SetEndTime();

    return (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
}


#endif


#ifdef __PLATFORM_WINDOWS__


void Timer::SetBeginTime()
{
    mTimerMask = 1;

     // Get the current process core mask
	DWORD procMask;
	DWORD sysMask;
#if _MSC_VER >= 1400 && defined (_M_X64)
	GetProcessAffinityMask(GetCurrentProcess(), (PDWORD_PTR)&procMask, (PDWORD_PTR)&sysMask);
#else
	GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask);
#endif

	// If procMask is 0, consider there is only one core available
	// (using 0 as procMask will cause an infinite loop below)
	if (procMask == 0)
		procMask = 1;

	// Find the lowest core that this process uses
	while( ( mTimerMask & procMask ) == 0 )
	{
		mTimerMask <<= 1;
	}

	HANDLE thread = GetCurrentThread();

	// Set affinity to the first core
	DWORD oldMask = SetThreadAffinityMask(thread, mTimerMask);

    // Get the constant frequency
    QueryPerformanceFrequency(&mFrequency);

	// Query the timer
    QueryPerformanceCounter(&begin);
	mStartTick = GetTickCount();

	// Reset affinity
	SetThreadAffinityMask(thread, oldMask);

	mLastTime = 0;
}

void Timer::SetEndTime()
{
    HANDLE thread = GetCurrentThread();

	// Set affinity to the first core
	DWORD oldMask = SetThreadAffinityMask(thread, mTimerMask);

	// Query the timer
    QueryPerformanceCounter(&end);

	// Reset affinity
	SetThreadAffinityMask(thread, oldMask);

    LONGLONG newTime = end.QuadPart - begin.QuadPart;
    
    // get milliseconds to check against GetTickCount
    unsigned long newTicks = (unsigned long) (1000 * newTime / mFrequency.QuadPart);
    
    // detect and compensate for performance counter leaps
    // (surprisingly common, see Microsoft KB: Q274323)
    unsigned long check = GetTickCount() - mStartTick;
    signed long msecOff = (signed long)(newTicks - check);
    if (msecOff < -100 || msecOff > 100)
    {
        // We must keep the timer running forward :)
        LONGLONG adjust = (std::min)(msecOff * mFrequency.QuadPart / 1000, newTime - mLastTime);
        begin.QuadPart += adjust;
        newTime -= adjust;
    }

    // Record last time for adjust
    mLastTime = newTime;
}

double Timer::GetElapsedSeconds()
{
    if (isRunning)
        SetEndTime();

    return static_cast<double>(mLastTime) / mFrequency.QuadPart;
}


#endif


#ifdef __PLATFORM_OTHER__


void Timer::SetBeginTime()
{
    begin = clock();
}

void Timer::SetEndTime()
{
    end = clock();
}

double Timer::GetElapsedSeconds()
{
    if (isRunning)
        SetEndTime();

    return static_cast<double>(end - begin) / CLOCKS_PER_SEC;
}


#endif


#endif