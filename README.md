## Quickscan
Quickscan is a document detection library for scanning and overlaying digital assets onto documents virtually. It uses the OpenCV library to detect document boundaries and skew them into an upright image. Thereafter, a Pygame window is used to draw on the upright image, and the drawings will be mapped back onto the original virtually.

[See it for yourself!](https://gfycat.com/naivesickfruitbat)

## Table of contents
- [Quick start](#quick-start)
- [What's included](#whats-included)
- [Documentation](#documentation)
- [Creators](#creators)
- [Thanks](#thanks)
- [Copyright and license](#copyright-and-license)

## Quick start
<a name="quick-start"></a>
This project is in its prototyping phase. You can get started with it by cloning the repo: `git clone https://github.com/shuiherng/quickscan.git`

## What's included
<a name="whats-included"></a>
The project files so far only include code which was used for the prototype. Sample images are not included, but you can place them in a `data` folder in the experiments directory.

## Documentation
<a name="documentation"></a>
The prototype is split into two distinct phases:
1. Document scanning phase
2. Input phase

### Document Scanning Phase
An image file specified in the notebook is fed through a document workflow, comprising:
1. Preprocessing Pipeline
2. Document Edge Detection
3. Page Extraction

The preprocessing pipeline utilizes a gaussian blur, denoiser, an Otsu thresholder, closer and edge detector. Kernel sizes are set by default to 3. Adjustments to the parameters for these filters may be required depending on the image used as input, especially if it is taken with a noisy background.

Document edge detection is performed by identifying [hough lines](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html), then clustering their intersections and taking their cluster means.

Page extraction is then done by figuring out the orientation of the page using this process:
Given a set of 4 x-y coordinates representing the edges of a document, 
1. The minimum x-y coordinate sum corresponds to the top-left point.
2. The maximum x-y coordinate sum cooresponds to the bottom-right point.
3. The remaining point which has a steeper gradient from the top-left point is the bottom-left point.
4. The last point is the top-right point.

This method allows landscape and portrait page inputs to be used, an improvement over the original algorithm proposed by Ishfar (see [thanks](#thanks)) which only works with portrait images.

### Input Phase
Inputs are accepted and processed using this workflow:
1. Draw input window
2. Read drawn inputs by user
3. Reverse mapping and overlay onto original image

The original image is scaled to match the height of the document. The original images and document are placed together on a pygame window, with a small separation (default: 20px). An invisible Pygame surface is drawn on top of the document. When inputs are given by the user via the mouse, the layer is used as input to a reverse transform which skews the drawing back to the original input image. This is then scaled down to match the size on the Pygame window, and drawn on the screen.

## Creators
<a name="creators"></a>

**Shui Herng Quek**
- <https://github.com/shuiherng>
- <https://www.linkedin.com/in/shui-herng-quek/>

## Thanks
<a name="thanks"></a>

Thanks to [Shakleen Ishfar](https://medium.com/@shakleenishfar) for writing the article on [Document Detection in Python](https://medium.com/intelligentmachines/document-detection-in-python-2f9ffd26bf65)! Some of the image preprocessing code in this project was based on his implementation.

## Copyright and license
<a name="copyright-and-license"></a>

Code and documentation copyright 2011–2022 the [Quickscan authors](https://github.com/shuiherng/quickscan/graphs/contributors). Code released under the [MIT License](https://github.com/shuiherng/quickscan/blob/main/LICENSE).