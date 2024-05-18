# Movie Recommendation System


https://github.com/Ibrahimmustafa1/MovieRecommendation/assets/85252957/34be936f-ec1e-4c0a-aa0f-b564057b1cf2


## Overview

This project is a Movie Recommendation System that uses advanced machine learning techniques to provide personalized movie recommendations. The system primarily employs content-based filtering, and we have experimented with Factorization Machines and Deep Learning for content-based filtering. Our findings indicate that Deep Learning achieved better results in comparison to Factorization Machines. The system is integrated into a Flask application, which exposes several API endpoints for fetching movie recommendations. Additionally, the backend server is integrated with an Angular application to provide a user-friendly interface.

## Features

- **Content-Based Filtering:** Recommends movies based on the content and attributes of movies.
- **Deep Learning Model:** Uses a neural network-based approach for better recommendation accuracy.
- **APIs:** Multiple endpoints for retrieving recommendations and similar movies.
- **Integration:** Backend server integrated with an Angular frontend for a seamless user experience.

## Dataset

The project utilizes the MovieLens dataset, which is a widely-used dataset in the field of movie recommendations.

## Implementation Details

### Models Used

1. **Factorization Machines:** Initially used for content-based filtering but found to be less effective compared to deep learning models.
2. **Deep Learning:** Achieved better results with a neural network-based approach. The model captures complex patterns and interactions in the data, providing more accurate recommendations.

### Flask Application

The backend of the system is built using Flask, a lightweight web framework for Python. The Flask app serves multiple API endpoints to interact with the recommendation system.

#### API Endpoints

1. **Get Movies by User ID**
   - **Endpoint:** `/rated_movies/<int:user_id>'`
   - **Method:** `GET`
   - **Description:** Fetches movie recommendations for a specific user based on their user ID.

2. **Get Recommendation for Specific User**
   - **Endpoint:** `/predict/<int:user_id>`
   - **Method:** `GET`
   - **Description:** Provides personalized movie recommendations using the saved deep learning model.

3. **Get Similar Movies**
   - **Endpoint:** `/similar/<int:item_id>`
   - **Method:** `GET`
   - **Description:** Retrieves movies similar to a given movie ID.

### Angular Application

The frontend of the system is built using Angular, a popular framework for building dynamic web applications. The Angular app interacts with the Flask backend through the provided API endpoints, offering users an intuitive interface to receive and view movie recommendations.

