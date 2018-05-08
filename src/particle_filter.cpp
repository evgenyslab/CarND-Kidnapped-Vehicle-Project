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
#include <limits> // added for DBL_MAX

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // set number of particles:
    num_particles = 20;
    // initilaize matrix sizes:
    particles.resize(num_particles);
    weights.resize(num_particles);
    // initialize generators:
    std::normal_distribution<double> x_dist(x,std[0]);
    std::normal_distribution<double> y_dist(y,std[1]);
    std::normal_distribution<double> theta_dist(theta,std[2]);
    // generate particles:
    for(auto&P : particles){
      P.x = x_dist(generator);
      P.y = y_dist(generator);
      P.theta = theta_dist(generator);
      P.weight = 1.0;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // initialize generators:
    std::normal_distribution<double> x_dist(0.0,std_pos[0]);
    std::normal_distribution<double> y_dist(0.0,std_pos[1]);
    std::normal_distribution<double> theta_dist(0.0,std_pos[2]);

  // iterate over each particle:
    for(auto&P : particles){
      // update position depending on yaw rate
      if (fabs(yaw_rate) < 1e-4) {
          P.x = P.x + (velocity * delta_t) * cos(P.theta) + x_dist(generator);
          P.y = P.y + (velocity * delta_t) * sin(P.theta) + y_dist(generator);
          P.theta = P.theta + theta_dist(generator);
      } else {
          P.x = P.x + (velocity / yaw_rate) * (sin(P.theta + yaw_rate * delta_t) - sin(P.theta)) + x_dist(generator);
          P.y = P.y + (velocity / yaw_rate) * (cos(P.theta) - cos(P.theta + yaw_rate * delta_t)) + y_dist(generator);
          P.theta = P.theta + yaw_rate * delta_t + theta_dist(generator);
      }
    }
}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> map_locations, std::vector<LandmarkObs>& observations) {

    // for each observation find nearest landmark, assign that landmark ID, this is computationally expensive.
      for(auto& Obs: observations){
        // set minimum distance to max value
        double dMin  = std::numeric_limits<double>::max();
            for(auto& MapLoc:map_locations){
            // calculate distance between observation and map location:
            double d = dist(Obs.x,Obs.y, MapLoc.x_f, MapLoc.y_f);
            // if distance is less than previous, store the match:
            if (d< dMin){
                // store current match
                Obs.id = MapLoc.id_i-1; // ensure the 0th index is taken.
                dMin = d;
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    // pre calculate constants:
    double s = 1.0/(2.0*M_PI*std_landmark[0] * std_landmark[1]);
    double d1 = 2.0 * pow(std_landmark[0], 2.0);
    double d2 = 2.0 * pow(std_landmark[1], 2.0);

    // each particle represents a possible vehicle position, speed & yaw rate, thus for each particle, need to test how
    // close the observations are to the map landmarks if the vehicle had the particle's state:
      for(auto& P: particles){
        // make a vector of map observations to store for this particle:
        std::vector<LandmarkObs> P_map_observations;
        // loop over each obeservation and convert it to map coordinates:
        for(auto& curr_observation: observations){
          LandmarkObs map_observation = LandmarkObs();
          // convert landmark observation to map coordinates:
          map_observation.x = P.x + (cos(P.theta) * curr_observation.x) - (sin(P.theta) * curr_observation.y);
          map_observation.y = P.y + (sin(P.theta) * curr_observation.x) + (cos(P.theta) * curr_observation.y);
          // do a sensor range check here, don't emplace if too far!
            (dist(P.x, P.y, map_observation.x, map_observation.y) > sensor_range? void(0):P_map_observations.emplace_back(map_observation));
        }
        // associate all transformed observation to the nearest landmarok on the map:
        dataAssociation(map_landmarks.landmark_list, P_map_observations);
        // reset all association statisitics:
        P.associations.clear();
        P.sense_x.clear();
        P.sense_y.clear();
        // initialize particle weight:
        double weight = 1.0;
        // Set the particle weight based on the distance sensed to the associated observations <-> map locations
        for (auto& Obs :P_map_observations){
          // local variable for current observation pose:
          double x_obs = Obs.x;
          double y_obs = Obs.y;
          // get ID of matched landmark to observation, and add it to association vector for paticle:
          P.associations.push_back(map_landmarks.landmark_list[Obs.id].id_i);
          // get x,y distance to landmark matched to observation:
          double landmark_x = map_landmarks.landmark_list[Obs.id].x_f;
          double landmark_y = map_landmarks.landmark_list[Obs.id].y_f;
          // push the landmark expected pose to the particle
          P.sense_x.push_back(landmark_x);
          P.sense_y.push_back(landmark_y);
          // update weight with bi-variate gaussian from observation to expected:
          weight *=  (s)*exp(-1*((pow((x_obs-landmark_x),2.0)/(d1))+(pow((y_obs-landmark_y),2.0)/(d2))));
        }
        // set particle weight:
        P.weight = weight;
    }

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
    weights.clear();
    for(auto& P : particles){
        weights.emplace_back(P.weight/w_sum);
    }

    // generate a discrete distribution:
    std::discrete_distribution<uint> dist(weights.begin(), weights.end());
    // create a new particle vector:
    std::vector<Particle> new_particles;
    new_particles.resize(num_particles);
    for(auto&P : new_particles){
      // draw from distribution:
      uint id = dist(generator);
      Particle newP = particles[id];
      // set new particle:
      P = newP;
    }
  // reset particles:
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
