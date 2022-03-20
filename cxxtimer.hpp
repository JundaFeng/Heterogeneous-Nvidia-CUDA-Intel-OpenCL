/*

MIT License

Copyright (c) 2017 André L. Maravilha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef CXX_TIMER_HPP
#define CXX_TIMER_HPP

#include <iostream>
#include <chrono>
#include <string>
#include <stack>
#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_RESET "\x1b[0m"

namespace cxxtimer {

/**
 * This class works as a stopwatch.
 */
class Timer {

public:

    /**
     * Constructor.
     *
     * @param   start
     *          If true, the timer is started just after construction.
     *          Otherwise, it will not be automatically started.
     */

    std::string message{""};
    std::string output_unit{""};

    Timer(const std::string msg, const std::string ounit) : message(msg), output_unit(ounit) {
            if (!started_) {
                started_ = true;
                paused_ = false;
                accumulated_ = std::chrono::duration<long double>(0);
                reference_ = std::chrono::steady_clock::now();
            } else if (paused_) {
                reference_ = std::chrono::steady_clock::now();
                paused_ = false;
            }
    };

    template <class duration_t = std::chrono::milliseconds>
    typename duration_t::rep log(){
        if (started_ && !paused_) {
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            accumulated_ = accumulated_ + std::chrono::duration_cast< std::chrono::duration<long double> >(now - reference_);
            paused_ = true;
        }
        auto t = std::chrono::duration_cast<duration_t>(accumulated_).count();
        const std::string to_print = message + " took " + ANSI_COLOR_GREEN + std::to_string(t) + " " + output_unit + ANSI_COLOR_RESET;
        std::cout << ANSI_COLOR_BLUE << "TIMER:: " << ANSI_COLOR_RESET << to_print <<std::endl;
        return t;
    }

    /**
     * Copy constructor.
     *
     * @param   other
     *          The object to be copied.
     */
    Timer(const Timer& other) = default;

    /**
     * Transfer constructor.
     *
     * @param   other
     *          The object to be transfered.
     */
    Timer(Timer&& other) = default;

    /**
     * Destructor.
     */
    virtual ~Timer() = default;

    /**
     * Assignment operator by copy.
     *
     * @param   other
     *          The object to be copied.
     *
     * @return  A reference to this object.
     */
    Timer& operator=(const Timer& other) = default;

    /**
     * Assignment operator by transfer.
     *
     * @param   other
     *          The object to be transferred.
     *
     * @return  A reference to this object.
     */
    Timer& operator=(Timer&& other) = default;

    /**
     * Reset the timer.
     */
    void reset() {
        if (started_) {
            started_ = false;
            paused_ = false;
            reference_ = std::chrono::steady_clock::now();
            accumulated_ = std::chrono::duration<long double>(0);
        }
    }
    /**
     * Return the elapsed time.
     *
     * @param   duration_t
     *          The duration type used to return the time elapsed. If not
     *          specified, it returns the time as represented by
     *          std::chrono::milliseconds.
     *
     * @return  The elapsed time.
     */


private:

    bool started_=false;
    bool paused_=false;
    std::chrono::steady_clock::time_point reference_;
    std::chrono::duration<long double> accumulated_;
};
    static std::stack<cxxtimer::Timer *> timer_table{};
}



static void timer_start(const std::string &msg, const char unit = 'm') {
    switch (unit) {
        case ' ':
            cxxtimer::timer_table.push(new cxxtimer::Timer(msg, "seconds"));
            break;
        case 'm':
            cxxtimer::timer_table.push(new cxxtimer::Timer(msg, "milliseconds"));
            break;
        case 'u':
            cxxtimer::timer_table.push(new cxxtimer::Timer(msg, "microseconds"));
            break;
        case 'n':
            cxxtimer::timer_table.push(new cxxtimer::Timer(msg, "nanoseconds"));
            break;
        default:
            cxxtimer::timer_table.push(new cxxtimer::Timer(msg, "minutes"));
            break;
    }
}

static void timer_stop(char unit = 'm') {
    /*
     *  ' ': second
     *  'm': millisecond
     *  so on so forth
     */
    auto entry = cxxtimer::timer_table.top();
    switch (unit) {
        case ' ':
            entry->log<std::chrono::seconds>();
            break;
        case 'm':
            entry->log<std::chrono::milliseconds>();
            break;
        case 'u':
            entry->log<std::chrono::microseconds>();
            break;
        case 'n':
            entry->log<std::chrono::nanoseconds>();
            break;
        default:
            entry->log<std::chrono::minutes>();
            break;
    }
    cxxtimer::timer_table.pop();
    delete entry;
}
#endif
