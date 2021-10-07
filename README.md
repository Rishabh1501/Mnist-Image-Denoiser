# Mnist-Image-Denoiser

Aim of this project is to remove noise from the Images using AutoEncoders. \
Deep Learning Framework: **Tensorflow**\
API Framework: **Flask**

<br>
  
## Features

- Responsive Web App
- Supports Every Operating System
- Can be easily deployed
- Easy to use architecture
 <br>
 
## Tech Stack

**Client:** HTML, CSS and Bootstrap

**Server:** Python

 <br>
 
## API Reference

### Train the Model

```http
  POST /train (type=json)
```

| Parameter (**Required**) | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `batch_size` | `int` |batch size  |
| `epochs` | `int` |Number of epochs to train |
| `model_save_path` | `path/string` |Folder/Directory name to store model  |
| `graph_save_path` | `path/string` |Path to Store the graph  |


### Prediction

```http
  POST /denoise (type=json)
```

| Parameter (**Required**) | Type     | Description                      |
| :-------- | :------- | :-------------------------------- |
| `img_path`      | `path/string` |  Path of the image |
| `model_path`      | `path/string` |  Location of the model |
| `output_save_path`      | `path/string` |  Path to save the output image |

<br>

## Installation

Install dependencies for the project using _**pip**_

```bash
    cd <project-folder>
    pip install requirements.txt
```
<br>

## Deployment

Project can be deployled on Heroku

```bash
  git add .
  git commit -m <message>
  git push heroku main
```
Remember to have the Buildpack included in settings.
```
https://github.com/heroku/heroku-buildpack-apt
```
Refer to the official documentation for more details: https://devcenter.heroku.com/articles/getting-started-with-python

<br>

## License

[MIT](https://choosealicense.com/licenses/mit/)

  
## Author

[@Rishabh1501](https://github.com/Rishabh1501)

  
## Support

For support, email at rishabhkalra1501@gmail.com .

  
