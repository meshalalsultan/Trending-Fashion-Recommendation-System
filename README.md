# Trending Fashion Recommendation System

## Overview
This project is a Trending Fashion Recommendation System built with a Flask web application. It provides product recommendations from the Myntra dataset based on product features like brand, ratings, price range, and popularity index. The recommendations are generated through a content-based filtering approach, ensuring users receive personalized product suggestions.

## Features
- Product recommendation based on similarity in ratings, price, and brand.
- Interactive web interface for users to input product ID and receive recommendations.
- Visualization of recommendation results in a user-friendly format.

## Installation

### Prerequisites
- Python 3.x
- Pip package manager

### Libraries
Use pip to install the following Python packages:


pip install flask pandas numpy scikit-learn


### Clone the Repository

git clone https://github.com/your-username/trending-fashion-recommendation-system.git

cd trending-fashion-recommendation-system


## Usage
1. Run the Flask app:


python app.py


2. Open a web browser and go to `http://127.0.0.1:5000/`.
3. Enter a Product ID in the input field and click 'Get Recommendations' to view similar products.

## How It Works
1. **Data Preprocessing**: The Myntra dataset is preprocessed to handle missing values and standardize the data.
2. **Feature Engineering**: Features like Popularity Index and Price Range are engineered from the dataset.
3. **Recommendation Logic**: Cosine similarity is used to compute the similarity between products based on their feature vectors.
4. **Web Interface**: Flask is used to create a web interface where users can input a Product ID and receive product recommendations.

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Linkedin - [www.linkedin.com/in/meshalhandal](www.linkedin.com/in/meshalhandal) - qualitymeshal@gmail.com

Website - [meshalalsultan.com]

Project Link: [https://github.com/your-username/trending-fashion-recommendation-system]

(https://github.com/your-username/trending-fashion-recommendation-system)

## Acknowledgements
- [Flask](https://flask.palletsprojects.com/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [SciKit-Learn](https://scikit-learn.org/)




