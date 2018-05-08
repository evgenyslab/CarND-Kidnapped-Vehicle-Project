/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // set number of particles:
    this->num_particles = 100;
    this->particles.resize(this->num_particles);
    this->weights.resize(this->num_particles);
    // initialize generators:
    std::normal_distribution<double> x_dist(x,std[0]);
    std::normal_distribution<double> y_dist(y,std[1]);
    std::normal_distribution<double> theta_dist(theta,std[2]);

    for(int i=0; i < this->num_particles; i++){
        // generate particle:
        Particle P;
        P.x = x_dist(generator);
        P.y = y_dist(generator);
        P.theta = theta_dist(generator);
        P.weight = 1.0;
        this->weights[i] = 1.0;
        // push particle into vector of particles:
        this->particles[i] = P;
    }
    std::cout << "Number of particles: " << this->particles.size() << "\n";
    for(auto& P : particles){
        std::cout << P.weight << "\n";
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // std_pose[] -> [x_o,y_o,theta_o]

    // x_f = x_o + (velocity/yaw_rate)*(sin(theta_o-yaw_rate*delta_t) - sin(theta_o)) + std_pos[0]
    // y_f = y_o + (velocity/yaw_rate)*(cos(theta_0)-cos(theta_o-yaw_rate*delta_t) + std_pos[1]
    // theta_f = theta_o + yaw_rate*delta_t + std_pos[2]

    // 0-mean gaussian noise:
    // std::default_random_engine generator;
    // std::normal_distribution<double> x_dist(0.0,0.1);

    std::default_random_engine generator;
    std::normal_distribution<double> x_dist(0.0,std_pos[0]);
    std::normal_distribution<double> y_dist(0.0,std_pos[1]);
    std::normal_distribution<double> theta_dist(0.0,std_pos[2]);
    for(int i=0; i < this->num_particles; i++){
        // generate particle:
        Particle P = this->particles[i];
        if (fabs(yaw_rate) < 1e-4) {
            P.x = P.x + (velocity * delta_t) * cos(P.theta) + x_dist(generator);
            P.y = P.y + (velocity * delta_t) * sin(P.theta) + y_dist(generator);
            P.theta = P.theta + theta_dist(generator);
        } else {
            P.x = P.x + (velocity / yaw_rate) * (sin(P.theta - yaw_rate * delta_t) - sin(P.theta)) + x_dist(generator);
            P.y = P.y + (velocity / yaw_rate) * (cos(P.theta) - cos(P.theta - yaw_rate * delta_t)) + y_dist(generator);
            P.theta = P.theta + yaw_rate * delta_t + theta_dist(generator);
        }
        // push particle into vector of particles:
        this->particles[i] = P;

    }
    is_initialized = true;

}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> map_locations, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    // for each observation find nearest landmark, assign that landmark ID, this is computationally expensive.
    for(int i=0; i< map_locations.size(); i++){
        double dMin  = 1e6;
        for(int j=0; j< observations.size(); j++){
            double d = dist(observations[j].x,observations[j].y, map_locations[i].x_f, map_locations[i].y_f);
            if (d< dMin){
                // store current match
                observations[j].id = i;
                dMin = d;
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // pre calc constants:
    double s = 1.0/(2.0*M_PI*std_landmark[0] * std_landmark[1]);
    double d1 = 2.0 * pow(std_landmark[0], 2.0);
    double d2 = 2.0 * pow(std_landmark[1], 2.0);

    // each particle represents a possible vehicle position, speed & yaw rate, thus for each particle, need to test how
    // close the observations are to the map landmarks
    for(int i = 0; i < particles.size(); i++){
        Particle P = particles[i];

        // transform observation into map coordinate:
        std::vector<LandmarkObs> P_map_observations;
        for (uint j = 0; j < observations.size(); j++) {
            LandmarkObs curr_observation = observations[j];
            LandmarkObs map_observation = LandmarkObs();
            // convert landmark observation to map coordinates:
            map_observation.x = P.x + (cos(P.theta) * curr_observation.x) - (sin(P.theta) * curr_observation.y);
            map_observation.y = P.y + (sin(P.theta) * curr_observation.x) + (cos(P.theta) * curr_observation.y);
            // do a sensor range check here, don't emplace if too far!
//            (dist(P.x, P.y, map_observation.x, map_observation.y) > sensor_range? void(0):P_map_observations.emplace_back(map_observation));
            P_map_observations.emplace_back(map_observation);
        }
        // associate all transformed observation to the nearest landmarok on the map:
        dataAssociation(map_landmarks.landmark_list, P_map_observations);
        particles[i].associations.clear();
        particles[i].sense_x.clear();
        particles[i].sense_y.clear();
        double weight = 1.0;
//        for (int j = 0; j < P_map_observations.size(); j++) {
        for (auto& d_observation :P_map_observations){

//            LandmarkObs d_observation = P_map_observations[j];
            double x_obs = d_observation.x;
            double y_obs = d_observation.y;
            // get ID of matched landmark to observation:
            particles[i].associations.push_back(map_landmarks.landmark_list[d_observation.id].id_i);
            // get x,y distance to landmark expected:
            double landmark_x = map_landmarks.landmark_list[d_observation.id].x_f;
            double landmark_y = map_landmarks.landmark_list[d_observation.id].y_f;
            // push the landmark expected pose to the particle
            particles[i].sense_x.push_back(landmark_x);
            particles[i].sense_y.push_back(landmark_y);
            // update weight with bi-variate gaussian from observation to expected:
            weight *=  (s)*exp(-1*((pow((x_obs-landmark_x),2.0)/(d1))+(pow((y_obs-landmark_y),2.0)/(d2))));
        }
        particles[i].weight = weight;
        std::cout << particles[i].weight << "\t";
    }
//    std::cout << "\n";

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // sum up all particle weights:
    double w_sum = 0;
    for(auto& P : particles){
        w_sum += P.weight;
    }
    // normalize particle weights:
    this->weights.clear();
    for(auto& P : particles){
        this->weights.emplace_back(P.weight/w_sum);
    }

    std::discrete_distribution<uint> dist(this->weights.begin(), this->weights.end());

    std::vector<Particle> new_particles;
    new_particles.resize(num_particles);
    for (int i = 0; i < num_particles; i++) {
        uint id = dist(generator);
        Particle newP = particles[id];
        new_particles[i] = newP;
    }
    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
