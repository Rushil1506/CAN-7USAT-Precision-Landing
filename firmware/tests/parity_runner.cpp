#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include "cansat/angles.hpp"
#include "cansat/pid.hpp"
#include "cansat/steering.hpp"
#include "cansat/servo_mapping.hpp"

int main() {
    std::cout << std::setprecision(12) << std::fixed;
    std::string line;

    cansat::PidConfig cfg;
    cansat::PidController pid(cfg);

    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "HEADING_ERROR") {
            double h, b;
            ss >> h >> b;
            double err = cansat::heading_error(h, b);
            std::cout << "HEADING_ERROR " << err << "\n";
        } else if (cmd == "BEARING") {
            double fx, fy, tx, ty;
            ss >> fx >> fy >> tx >> ty;
            double brg = cansat::bearing(fx, fy, tx, ty);
            std::cout << "BEARING " << brg << "\n";
        } else if (cmd == "PID_INIT") {
            ss >> cfg.kp >> cfg.ki >> cfg.kd >> cfg.output_min >> cfg.output_max >> cfg.integral_min >> cfg.integral_max;
            pid.set_config(cfg);
            pid.reset();
            std::cout << "PID_INIT OK\n";
        } else if (cmd == "PID_RESET") {
            pid.reset();
            std::cout << "PID_RESET OK\n";
        } else if (cmd == "PID_UPDATE") {
            double err, dt;
            ss >> err >> dt;
            double out = 0.0;
            cansat::PidStatus stat = pid.update(err, dt, out);
            std::cout << "PID_UPDATE " << static_cast<int>(stat) << " " << out << " " << pid.integral() << " " << pid.previous_error() << "\n";
        } else if (cmd == "STEER") {
            double s;
            ss >> s;
            cansat::BrakeCommand b = cansat::steering_to_brakes(s);
            std::cout << "STEER " << b.left << " " << b.right << "\n";
        } else if (cmd == "SERVO") {
            double bl, br, ln, lm, rn, rm;
            ss >> bl >> br >> ln >> lm >> rn >> rm;
            cansat::BrakeCommand b{bl, br};
            cansat::ServoConfig lc{ln, lm};
            cansat::ServoConfig rc{rn, rm};
            cansat::ServoCommand sc = cansat::brake_to_servo(b, lc, rc);
            std::cout << "SERVO " << sc.left_angle << " " << sc.right_angle << "\n";
        }
    }
    return 0;
}
