# I2I Master

## About the Project

This project aims to do the face expression exchange between animals.

We have implemented in two ways.

The CNN + TPS + FUSION:
Currently this can only do face expression exchange between cats.
![cnn screenshot](cnn.png)

The GAN:
Currently this can only do face expression between Japanese Spitz and French Bulldog.
![gan screenshot](gan.png)

### Built With

* [React.js](https://reactjs.org/)
* [Flask](https://flask.palletsprojects.com/en/2.0.x/)
* [keras](https://keras.io/)

## Getting Started

### Prerequisites

* Node v16.8.0
* Flask
* tensorflow 2.6.0
* keras-contrib

    ```sh
    pip install git+https://www.github.com/keras-team/keras-contrib.git`
    ```

* PIL
* OpenCV-Python

### Installation

1. Git clone this repo

2. Start backend server \
`cd backend` \
`python3 chatbot.py`

3. Start frontend \
`cd frontend` \
`npm install` (only for first time) \
`npm start`

## Usage

Select the mode first, `CNN+TPS+FUSION` or `GAN`.

Then just drag or click the upload button to upload the image, and then click `Translate`.

## Repo Files Structure

* `backend` folder contains the backend code
* `frontend` folder contains the frontend code
* `images` folder contains some sample images that can upload to the website
* `CNN` folder contains the process of marking the images, trying how TPS works and training the CNN models

## Acknowledgments

This is the project for Pattern Recognition Systems (PRS), which is offered by NUS-ISS.

## Performance

![image](https://user-images.githubusercontent.com/88467925/143771615-d02287a9-49b8-4770-a189-d089b9049aff.png)
![image](https://user-images.githubusercontent.com/88467925/143771645-48e68615-a2a3-4656-a344-1b3992e75de8.png)
![image](https://user-images.githubusercontent.com/88467925/143771669-2f307860-5ee3-4dbd-9c81-0dcd1e24bde3.png)



