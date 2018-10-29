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
	
	num_particles = 15;	
	default_random_engine gen;
	normal_distribution<double> init_x(0, std[0]);
	normal_distribution<double> init_y(0, std[1]);
	normal_distribution<double> init_theta(0, std[2]);
	
	for (int i=0; i<num_particles; i++)
	{
		Particle part;
		part.id = i;
		part.x = x+init_x(gen);
		part.y = y+init_y(gen);
		part.theta = theta+init_theta(gen);
		part.weight = 1.0;
		particles.push_back(part);
		weights.push_back(1.0);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);
	
	for (int i=0; i<num_particles; i++)
	{
		double x0 = particles[i].x;
		double y0 = particles[i].y;
		double theta0 = particles[i].theta;
		// CTRV equations from Lesson 7: #20. Sigma Point Prediction Assignment 1
		if (fabs(yaw_rate) < 0.00001)
		{
			x0 += velocity*cos(theta0)*delta_t;
			y0 += velocity*sin(theta0)*delta_t;
		}
		else
		{
			x0 += velocity/yaw_rate*(sin(theta0+yaw_rate*delta_t)-sin(theta0));
			y0 += velocity/yaw_rate*(cos(theta0)-cos(theta0+yaw_rate*delta_t));
			theta0 += yaw_rate*delta_t;
		}
		
		normal_distribution<double> dist_x(x0, std_pos[0]);
		normal_distribution<double> dist_y(y0, std_pos[1]);
		normal_distribution<double> dist_theta(theta0, std_pos[2]);
	
		particles[i].x = dist_x(gen)+N_x(gen);
		particles[i].y = dist_y(gen)+N_y(gen);
		particles[i].theta = dist_theta(gen)+N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (unsigned int i=0; i<observations.size(); i++)
	{
		double min_dist = numeric_limits<double>::max(); // Initialized to some high unattainable value
		//int j_true;
		int landmark_id = -1;
		for (unsigned int j=0; j<predicted.size(); j++)
		{
			double obs_dist = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (obs_dist < min_dist)
			{
				min_dist = obs_dist;
				landmark_id = predicted[j].id;
				//j_true = j;
			}
		}
		//cout << "Observation " << observations[i].id << " coordinates: " << observations[i].x << "\t" << observations[i].y << endl;
		observations[i].id = landmark_id;
		//cout << " Landmark " << predicted[j_true].id << " coordinates: " << predicted[j_true].x << "\t" << predicted[j_true].y << endl;
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
	
	// map_landmarks.landmark_list is the list of landmark coordinates in the MAP coordinate system
	
	for (int i=0; i<num_particles; i++)
	{
		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double theta_p = particles[i].theta;
		cout << "Particle " << i << ":" << endl;
		cout << "Particle coordinates: " << x_p << "\t" << y_p << endl;
		
		// Separate out landmarks only within sensor range of the particle
		vector<LandmarkObs> landmarks_in_range;
		for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++)
		{
			float x_lm = map_landmarks.landmark_list[j].x_f;
			float y_lm = map_landmarks.landmark_list[j].y_f;
			double landmark_dist = dist(x_lm, y_lm, x_p, y_p);
			if (landmark_dist < sensor_range)
			{
				landmarks_in_range.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i,x_lm,y_lm});
				//cout << "Landmark " << map_landmarks.landmark_list[j].id_i << " coordinates: " << x_lm << "\t" << y_lm << endl;
			}
		}
		//cout << "Landmarks in range size: " << landmarks_in_range.size() << endl;
		
		// Transform observations from the particle location from the vehicle to map coordinate system
		vector<LandmarkObs> transformed_observations;
		for (unsigned int j=0; j<observations.size(); j++)
		{
			double x_t = x_p + observations[j].x*cos(theta_p) - observations[j].y*sin(theta_p);
			double y_t = y_p + observations[j].x*sin(theta_p) + observations[j].y*cos(theta_p);
			transformed_observations.push_back(LandmarkObs{observations[j].id,x_t,y_t});
			//cout << "Observation " << observations[j].id << " (" << observations[j].x << "," << observations[j].y << ") -> (" << transformed_observations[j].x << "," << transformed_observations[j].y << ")" << endl;
		}
		//cout << "Transformed observations size: " << transformed_observations.size() << endl;
		
		// Associate transformed observations with landmarks
		dataAssociation(landmarks_in_range, transformed_observations);
		
		// Calculate the particle's final weight
		double sigma_x = std_landmark[0];
		double sigma_y = std_landmark[1];
		particles[i].weight = 1.0;
		for (unsigned int j=0; j<transformed_observations.size(); j++)
		{
			double mu_x, mu_y, x_lir, y_lir;
			mu_x = transformed_observations[j].x;
			mu_y = transformed_observations[j].y;
			//int id = transformed_observations[j].id;
			for (unsigned int k=0; k<landmarks_in_range.size(); k++)
			{
				if (landmarks_in_range[k].id == transformed_observations[j].id)
				{
					//cout << "Transformed observation ID: " << transformed_observations[j].id << endl;
					x_lir = landmarks_in_range[k].x;
					y_lir = landmarks_in_range[k].y;
					//cout << "Landmark coordinates: " << x_lir << "\t" << y_lir << endl;
				}
			}
			//x_lir = map_landmarks.landmark_list[id-1].x_f;
			//y_lir = map_landmarks.landmark_list[id-1].y_f;
			double obs_weight = (1/(2*M_PI*sigma_x*sigma_y)) * exp(-(pow(x_lir-mu_x,2)/(2*pow(sigma_x,2)) + (pow(y_lir-mu_y,2)/(2*pow(sigma_y,2)))));
			cout << "Landmark " << transformed_observations[j].id << " weight: " << obs_weight << endl;
			particles[i].weight *= obs_weight;
		}
		weights.push_back(particles[i].weight);
		cout << "Particle " << i << " weight: " << particles[i].weight << "\n" << endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	// Code borrowed from https://github.com/jeremy-shannon/CarND-Kidnapped-Vehicle-Project
	
	default_random_engine gen;
	vector<Particle> new_particles;

	// get all of the current weights
	vector<double> weights;
	for (int i = 0; i < num_particles; i++)
	{
		weights.push_back(particles[i].weight);
	}

	// generate random starting index for resampling wheel
	uniform_int_distribution<int> uniintdist(0, num_particles-1);
	auto index = uniintdist(gen);

	// get max weight
	double max_weight = *max_element(weights.begin(), weights.end());

	// uniform random distribution [0.0, max_weight)
	uniform_real_distribution<double> unirealdist(0.0, max_weight);

	double beta = 0.0;

	// spin the resample wheel!
	for (int i = 0; i < num_particles; i++)
	{
		beta += unirealdist(gen) * 2.0;
		while (beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
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
	return particle;
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
