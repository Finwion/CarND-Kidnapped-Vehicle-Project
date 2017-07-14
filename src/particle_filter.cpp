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

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[3]);
	default_random_engine gen;

	for (int i = 0; i < num_particles; ++i)
	{
		Particle sample_particle;

		sample_particle.id = i;
		sample_particle.x = dist_x(gen);
		sample_particle.y = dist_y(gen);
		sample_particle.theta = dist_theta(gen);
		sample_particle.weight = 1.0;

		particles.push_back(sample_particle);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for (int i = 0; i < num_particles; ++i)
	{
		if (fabs(yaw_rate) < 1e-6)
		{
			particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
		}
		else
		{
			particles[i].x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta = particles[i].theta + yaw_rate * delta_t;
		}
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[3]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); ++i)
	{
		double minimum_distance = std::numeric_limits<double>::max();
		for (int j = 0; j < predicted.size(); ++j)
		{
			//Calculate Euclidean distance between observed measurement and predicted measurement
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (distance < minimum_distance)
			{
				observations[i].id = predicted[j].id; //assign observed measurement ID to this landmark if closer than previous
				minimum_distance = distance;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   std::vector<LandmarkObs> observations, Map map_landmarks)
{
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for (int i = 0; i < num_particles; ++i)
	{
		vector<LandmarkObs> gbl_observations; //Keep track of observations in map or global coordinate space
		vector<LandmarkObs> filtered_landmarks;

		particles[i].weight = 1.0;
		//translate observations from the particle/car coordinate space to global/map coordinate space
		for (int j = 0; j < observations.size(); ++j)
		{
			LandmarkObs curr_obj;
			curr_obj.x = observations[j].x * cos(particles[j].theta) - observations[j].y * sin(particles[j].theta) + particles[i].x;
			curr_obj.y = observations[j].x * sin(particles[j].theta) + observations[j].y * cos(particles[j].theta) + particles[i].y;
			curr_obj.id = -1;
			gbl_observations.push_back(curr_obj);
		}

		// Filter landmarks that are not in range of the particle
		for (int j = 0; j <= map_landmarks.landmark_list.size(); ++j)
		{
			if (dist(particles[i].x, particles[i].y,
					 map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) <= sensor_range)
			{
				LandmarkObs curr_landmark;
				curr_landmark = {map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f};
				filtered_landmarks.push_back(curr_landmark);
			}
		}

		//nearest neighbor association to match observed measurement to the landmark
		// and check to see if nearest neighbor is in range
		dataAssociation(filtered_landmarks, gbl_observations);

		for (int j = 0; j < gbl_observations.size(); ++j)
		{
			double mu_x;
			double mu_y;
			for (int k = 0; k < filtered_landmarks.size(); ++k)
			{

				if (filtered_landmarks[k].id == gbl_observations[j].id)
				{
					mu_x = filtered_landmarks[k].x;
					mu_y = filtered_landmarks[k].y;
				}
			}
			//Update weights using multi-variate Gaussian distribution
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double var_x = pow(std_x, 2);
			double var_y = pow(std_y, 2);
			double x_num = pow(gbl_observations[j].x - mu_x, 2);
			double y_num = pow(gbl_observations[j].y - mu_y, 2);

			particles[i].weight *= exp(-x_num / (2 * var_x) - y_num / (2 * var_y)) / (2 * M_PI * std_x * std_y);
		}
		weights[i] = particles[i].weight; 
	}
}

void ParticleFilter::resample()
{
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	vector<Particle> resampled_particles;
	//discrete distribution constructed from a list of particle weights 
	discrete_distribution<int> weight_dist(weights.begin(), weights.end());

	for (int i = 0; i < particles.size(); ++i)
	{
		particles[i] = particles[weight_dist(gen)];
	}
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
